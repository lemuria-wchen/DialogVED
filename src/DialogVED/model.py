import math
import random
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils

from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
    PositionalEmbedding
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .attention import (
    RelativePositionBias,
    MultiheadAttentionRPE,
    NgramMultiheadAttention,
    ngram_attention_bias,
    LearnedPositionalEmbeddingNew
)


@register_model('ngram_transformer_prophet')
class NgramTransformerProphetModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        # --------------------------------------------------------------------------------------------------------------
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N', help='num decoder attention heads')
        parser.add_argument('--encoder-layer-drop', type=float, default=0.0)

        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--ngram', type=int, metavar='N',
                            help='num of predicting grams')
        parser.add_argument('--num_buckets', type=int, metavar='N',
                            help='num of buckets for relative position')
        parser.add_argument('--relative_max_distance', type=int, metavar='N',
                            help='num of bucket for relative position')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings '
                                 '(requires shared dictionary and embed dim)')
        # --------------------------------------------------------------------------------------------------------------
        parser.add_argument('--with-mask-lm-logits', action='store_true',
                            help='whether to include masked language module')
        parser.add_argument('--with-cls-bow-logits', action='store_true',
                            help='whether to include [CLS] bag-of-word logits')
        parser.add_argument('--with-latent-bow-logits', action='store_true',
                            help='whether to include latent bag-of-word logits')
        parser.add_argument('--extend-latent-with-cls', action='store_true',
                            help='whether to extend latent variable with [CLS] feature')

        parser.add_argument('--disable-kl-loss', action='store_true',
                            help='whether to disable kullback–leibler divergence loss')

        parser.add_argument('--with-encoder-ape-token', action='store_true')
        parser.add_argument('--with-encoder-ape-sentence', action='store_true')
        parser.add_argument('--with-encoder-ape-role', action='store_true')

        parser.add_argument('--with-encoder-rpe-token', action='store_true')
        parser.add_argument('--with-encoder-rpe-sentence', action='store_true')

        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='whether to generate deterministic latent variable')

        parser.add_argument('--target-kl', default=3.0, type=float, help='target k-l loss')
        parser.add_argument('--kl-loss-weight', default=1.0, type=float, help='kl divergence loss weight')
        parser.add_argument('--cls-bow-loss-weight', default=0.5, type=float, help='bag of word loss weight')
        parser.add_argument('--latent-bow-loss-weight', default=0.5, type=float, help='bag of word loss weight')
        parser.add_argument('--masked-lm-loss-weight', default=1.0, type=float, help='mask lm loss weight')

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    @staticmethod
    def print_args(args):
        # parameters of interest
        iargs = [
            # task specific
            'mask_source',
            'add_cls_to_source',
            'generate_latent_variable',
            # logits
            'with_mask_lm_logits',
            'with_cls_bow_logits',
            'with_latent_bow_logits',
            'extend_latent_with_cls',
            'use_latent_variable',
            'disable_kl_loss',
            # loss weight
            'masked_lm_loss_weight',
            'cls_bow_loss_weight',
            'latent_bow_loss_weight',
            'kl_loss_weight',
            # position embedding
            'with_encoder_ape_sentence',
            'with_encoder_ape_role',
            'with_encoder_rpe_token',
            'with_encoder_rpe_sentence',
            'deterministic',
            'latent_size',
        ]
        print('='.ljust(66, '='))
        for arg in vars(args):
            if arg in iargs:
                print("{} = {}".format(arg, getattr(args, arg)))
        print('-'.ljust(66, '-'))
        print('| Time stamp: {}'.format(str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))))

    @staticmethod
    def check_args(args):
        print('-'.ljust(66, '-'))
        if args.mask_source and not args.with_mask_lm_logits:
            warnings.warn(message='language span masked but with not masked language loss !')
        if args.with_mask_lm_logits:
            # assert args.mask_source
            if not args.masked_lm_loss_weight > 0.0:
                warnings.warn(message='masked lm logits computed but with not positive loss weight !')
        if args.generate_latent_variable:
            if not args.with_latent_bow_logits:
                warnings.warn(message='latent variable is generated but bag-of-word loss is not optimized !')
            if not args.add_cls_to_source:
                warnings.warn(message='latent variable is generated by other tokens but not [CLS] !')
            if not args.use_latent_variable:
                warnings.warn(message='latent variable is generated but not used !')
        if args.with_cls_bow_logits and not args.add_cls_to_source:
            warnings.warn(message='cls bag-of-word logits is generated by other tokens but not [CLS] !')
        if args.with_cls_bow_logits and not args.cls_bow_loss_weight > 0.0:
            warnings.warn(message='cls bag-of-word logits is generated but with not positive loss weight !')
        if args.extend_latent_with_cls and not args.add_cls_to_source:
            warnings.warn(message='latent variable is generated by other tokens but not [CLS] !')
        if args.with_latent_bow_logits and not args.add_cls_to_source:
            warnings.warn(message='latent bag-of-word logits is generated by other tokens but not [CLS] !')
        if args.with_latent_bow_logits and not args.latent_bow_loss_weight > 0.0:
            warnings.warn(message='latent bag-of-word logits is generated but with not positive loss weight !')
        if args.disable_kl_loss and args.kl_loss_weight > 0.0:
            warnings.warn(message='k-l divergence loss is disabled but with positive k-l loss weight !')
        if args.with_encoder_ape_sentence or args.with_encoder_ape_role:
            assert args.auto_infer_absolute_positions
        if args.with_encoder_rpe_token or args.with_encoder_rpe_sentence:
            assert args.auto_infer_relative_positions
        print('-'.ljust(66, '-'))

    @classmethod
    def build_model(cls, args, task):

        # load default model parameters / this function can be applied to all modules in fairseq
        # task/model/criterion etc.
        base_architecture(args)

        # check args
        # cls.check_args(args)

        # print args
        # cls.print_args(args)

        # source and target dictionary, in most translation task, dictionary is different
        # but in a typical english/chinese text-to-text or data-to-text task, dictionary is joined
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            # build Embedding layer in Transformer Encoder & decoder
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = encoder_embed_tokens
            # share parameters in bag_of_word predicted layer in VAE
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        # if share_all_embeddings, parameters of this module will not be trained
        bow_embed_tokens_enc = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)
        bow_embed_tokens_latent = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)
        mask_embed_tokens = nn.Linear(in_features=args.encoder_embed_dim, out_features=len(tgt_dict), bias=False)

        # initial encoder and decoder
        encoder = TransformerEncoder(
            args, src_dict, encoder_embed_tokens, bow_embed_tokens_enc, bow_embed_tokens_latent, mask_embed_tokens)
        decoder = NgramTransformerDecoder(args, tgt_dict, decoder_embed_tokens)

        model = NgramTransformerProphetModel(encoder, decoder)

        if args.load_from_pretrained_model is not None:
            print('loading pretrained model from {}'.format(args.load_from_pretrained_model))
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            if 'model' in states and 'args' in states:
                states = states['model']
            # replaced = {
            #     'encoder.embed_positions.weight': 'encoder.embed_ap_token.weight',
            #     'decoder.embed_positions.weight': 'decoder.embed_ap_token.weight',
            # }
            # for key in replaced:
            #     if key in states:
            #         _, _embed_dim = states[key].shape
            #         _dtype = states[key].dtype
            #         states[replaced.get(key)] = torch.cat(
            #             [torch.zeros(1, _embed_dim, dtype=_dtype), states[key]], dim=0)
            #         del states[key]
            # adapt to new positions setting
            # try:
            #     for position_name, target_position_length in [
            #         ('encoder.embed_ap_token.weight', model.encoder.embed_ap_token.weight.size(0)),
            #         ('decoder.embed_ap_token.weight', model.decoder.embed_ap_token.weight.size(0)),
            #     ]:
            #         if states[position_name].size(0) < target_position_length:
            #             _index = torch.arange(states[position_name].size(1))
            #             expend_position_states = states[position_name].clone()
            #             while states[position_name].size(0) < target_position_length:
            #                 _index = torch.cat((_index[1:], _index[:1]), dim=0)
            #                 states[position_name] = torch.cat([
            #                     states[position_name], expend_position_states[:, _index]], dim=0)
            #         if states[position_name].size(0) > target_position_length:
            #             states[position_name] = states[position_name][:target_position_length]
            # except (AttributeError, KeyError):
            #     pass
            # # delete unmatched keys
            # unmatched_keys = ['encoder.vae_fc3.weight', 'decoder.vae_transform.weight', 'decoder.vae_transform.bias']
            # for key in unmatched_keys:
            #     if key in states:
            #         del states[key]

            # adapt to new positions setting
            try:
                for position_name, target_position_length in [
                    ('encoder.embed_positions_token.weight', model.encoder.embed_ap_token.weight.size(0)),
                    ('decoder.embed_positions_token.weight', model.decoder.embed_ap_token.weight.size(0)),
                ]:
                    if states[position_name].size(0) < target_position_length:
                        _index = torch.arange(states[position_name].size(1))
                        expend_position_states = states[position_name].clone()
                        while states[position_name].size(0) < target_position_length:
                            _index = torch.cat((_index[1:], _index[:1]), dim=0)
                            states[position_name] = torch.cat([
                                states[position_name], expend_position_states[:, _index]], dim=0)
                    if states[position_name].size(0) > target_position_length:
                        states[position_name] = states[position_name][:target_position_length]
            except (AttributeError, KeyError):
                pass
            # load pre-trained layers
            model_dict = model.state_dict()
            # compare to ProphetNet, what is updated
            print('| new in current model: ')
            print([k for k, v in model_dict.items() if k not in states.keys()])
            print('| discard from original model: ')
            print([k for k, v in states.items() if k not in model_dict.keys()])
            state_dict = {k: v for k, v in states.items() if k in model_dict.keys()}
            print('| updating parameters ...')
            print('-'.ljust(66, '-'))
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            args.load_from_pretrained_model = None

        return NgramTransformerProphetModel(encoder, decoder)

    def max_positions(self):
        return self.encoder.max_positions(), self.decoder.max_positions()

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, **kwargs):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)

        return decoder_out, encoder_out


# initialize torch.nn.Embedding with normal distribution and zero constant padding
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


# initialize torch.nn.Linear with xavier uniform weight and zero bias
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


# build embedding based on dictionary
def build_embedding(dictionary, embed_dim):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb


# Transformer Encoder Layer
# original paper method: dropout -> add residual -> layer_norm
# multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
# position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
#
# tensor2tensor method: layer_norm -> dropout -> add residual
# multi-head self-attention sublayer: x = x + dropout(self_attn(layer_norm(x))))
# position-wise feed-forward sublayer: x = x + dropout(fc2(dropout(relu(fc1(layer_norm(x))))))
# in this paper we adopt the original structure


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self, encoder_embed_dim, encoder_ffn_embed_dim, encoder_attention_heads, dropout,
            attention_dropout, activation_dropout, activation_fn,
            embed_relative_positions_token=None, embed_relative_positions_sentence=None,
    ):
        super().__init__()
        self.embed_dim = encoder_embed_dim
        """
            compare to `MultiheadAttention`, `MultiheadAttentionRPE` is equipped with relative position embedding
            if both the embed_relative_positions_token and embed_relative_positions_sentence is None
            this module is completely the same as as MultiheadAttention
        """
        self.embed_relative_positions_token = embed_relative_positions_token
        self.embed_relative_positions_sentence = embed_relative_positions_sentence
        if embed_relative_positions_token is None and embed_relative_positions_sentence is None:
            self.self_attn = MultiheadAttention(
                self.embed_dim, encoder_attention_heads,
                dropout=attention_dropout, self_attention=True,
            )
        else:
            self.self_attn = MultiheadAttentionRPE(
                self.embed_dim, encoder_attention_heads,
                dropout=attention_dropout, self_attention=True,
                embed_relative_positions_token=embed_relative_positions_token,
                embed_relative_positions_sentence=embed_relative_positions_sentence,
            )
        # self.self_attn = MultiheadAttentionRPE(
        #     self.embed_dim, encoder_attention_heads,
        #     dropout=attention_dropout, self_attention=True,
        #     embed_relative_positions_token=embed_relative_positions_token,
        #     embed_relative_positions_sentence=embed_relative_positions_sentence,
        # )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask=None, relative_position_token=None, relative_position_sentence=None):
        """
        Args:
            x: input to the t-layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask: binary ByteTensor of shape `(batch, src_len)`
            relative_position_token: token relative position tensor of shape `(batch, src_len, src_len)`
            relative_position_sentence: token relative position tensor of shape `(batch, src_len, src_len)`
        """
        # Note:
        # `key_padding_mask` is usually not None, since text sequences have variable length
        # `attn_mask` is usually None, unless some tokens do not need to attend to other tokens
        # but it is not None in decoder layer

        # multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
        residual = x
        """
            Note: the token between different distances should have different patterns of attention
            even if the distance between tokens is the same, but the distance between sentences is different
            there may be also different attention patterns
        """

        # x, _ = self.self_attn(
        #     query=x, key=x, value=x,
        #     key_padding_mask=encoder_padding_mask, need_weights=False,
        #     # relative position embedding (PE) here
        #     # relative token PE + relative sentence PE + relative token PE * relative sentence PE
        #     relative_position_token=relative_position_token,
        #     relative_position_sentence=relative_position_sentence,
        # )
        if self.embed_relative_positions_token is None and self.embed_relative_positions_token is None:
            x, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask, need_weights=False,
            )
        else:
            x, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=encoder_padding_mask, need_weights=False,
                # relative position embedding (PE) here
                # relative token PE + relative sentence PE + relative token PE * relative sentence PE
                relative_position_token=relative_position_token,
                relative_position_sentence=relative_position_sentence,
            )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        # position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x


class TransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, bow_embed_tokens_enc=None,
                 bow_embed_tokens_latent=None, mask_embed_tokens=None):
        super().__init__(dictionary)

        self.dictionary = dictionary
        self.dropout = args.dropout
        self.encoder_layer_drop = args.encoder_layer_drop

        # absolute position embedding setting
        self.with_encoder_ape_token = args.with_encoder_ape_token
        self.with_encoder_ape_sentence = args.with_encoder_ape_sentence
        self.with_encoder_ape_role = args.with_encoder_ape_role

        # relative position embedding setting
        self.with_encoder_rpe_token = args.with_encoder_rpe_token
        self.with_encoder_rpe_sentence = args.with_encoder_rpe_sentence

        # training mode is seq2seq or vae
        self.generate_latent_variable = args.generate_latent_variable
        # map the [CLS] token hidden to feature to predict bag of word
        self.with_cls_bow_logits = args.with_cls_bow_logits
        # map the latent variable to feature to predict bag of word
        self.with_latent_bow_logits = args.with_latent_bow_logits
        # map the masked hidden to feature to predict mask language
        self.with_mask_lm_logits = args.with_mask_lm_logits
        # concatenate the latent variable with encoder feature to predict bag of word logits
        self.extend_latent_with_cls = args.extend_latent_with_cls

        # variational auto-encoder setting
        # self.target_kl = args.target_kl if self.generate_latent_variable else None
        # self.deterministic = args.deterministic if not self.generate_latent_variable else None
        # self.disable_kl_loss = args.disable_kl_loss if not self.generate_latent_variable else None
        self.deterministic = getattr(args, 'deterministic', False)
        self.disable_kl_loss = getattr(args, 'disable_kl_loss', False)

        self.embed_dim = args.encoder_embed_dim
        assert embed_tokens.embedding_dim == self.embed_dim, \
            '`encoder_embed_dim` parameter in global args must equal to the one in self-attention'

        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.share_all_embeddings = args.share_all_embeddings

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)

        # absolute position embedding (APE) -> cite PLATO here < https://arxiv.org/pdf/1910.07931.pdf >

        # token PE (default padding index is 1)
        if self.with_encoder_ape_token:
            self.embed_ap_token = PositionalEmbedding(
                args.max_source_positions_token, self.embed_dim, self.padding_idx, learned=True)
        else:
            self.embed_ap_token = None
            warnings.warn(message='there is a risk in using this setting because '
                                  'the transformer model is order insensitive !')

        # sentence PE (padding index is 0)
        """
            response sentence is supposed to always be sentence 1, so if the dialogue is: 
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
            then the sentence index is:
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
               5        4         3        2        1
            since the response is always sentence 1, we do not need to build similar module in decoder
        """
        self.embed_ap_sentence = PositionalEmbedding(
            args.max_source_positions_sentence, self.embed_dim, padding_idx=None, learned=True,
        ) if self.with_encoder_ape_sentence else None

        if self.embed_ap_sentence is not None:
            nn.init.constant_(self.embed_ap_sentence.weight[0], 0)

        # role PE (padding index is 0)
        """
            response role is supposed to always be role 1, so if the dialogue is: 
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
            then the role index is:
            <turn 1> <turn 2> <turn 3> <turn 4> <response>
               1        2         1        2        1
            
            if there is additional knowledge in the input, it should be placed at the beginning
            and additional knowledge should be assigned one index
            <knowledge 1> <knowledge 2> <knowledge 3> <turn 1> <turn 2> <turn 3> <turn 4> <response>
                 3              3            3           1        2        1         2         1
                 
            since the response is always role 1, we do not need to build similar module in decoder
        """
        self.embed_ap_role = PositionalEmbedding(
            args.max_source_positions_role, self.embed_dim, padding_idx=None, learned=True,
        ) if self.with_encoder_ape_role else None

        if self.embed_ap_role is not None:
            nn.init.constant_(self.embed_ap_role.weight[0], 0)

        # relative position embedding (RPE) -> cite T5 & ProphetNet here
        self.embed_rp_token = RelativePositionBias(
            embed_dim=self.embed_dim,
            num_buckets=args.num_buckets_source_token,
            max_distance=args.max_distance_source_token,
            n_heads=args.encoder_attention_heads,
            bidirectional=args.bidirectional_source_token,
        ) if self.with_encoder_rpe_token else None

        self.embed_rp_sentence = RelativePositionBias(
            embed_dim=self.embed_dim,
            num_buckets=args.num_buckets_source_sentence,
            max_distance=args.max_distance_source_sentence,
            n_heads=args.encoder_attention_heads,
            bidirectional=args.bidirectional_source_sentence,
        ) if self.with_encoder_rpe_sentence else None

        self.layers = nn.ModuleList([])
        # embed_relative_positions layer are shared in all encoder layers
        self.layers.extend([
            TransformerEncoderLayer(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
                self.embed_rp_token,
                self.embed_rp_sentence,
            )
            for _ in range(args.encoder_layers)
        ])

        # self.emb_layer_norm = LayerNorm(self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)

        if self.with_cls_bow_logits:
            # use vae module first linear layer
            self.vae_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            # bag of words predictor linear layer
            self.bow_fc_enc = bow_embed_tokens_enc if not self.share_all_embeddings else None
        else:
            self.vae_fc1 = None

        if self.generate_latent_variable:
            # if no encoder decoder attention, it's variational auto-encoder mode
            if self.vae_fc1 is None:
                self.vae_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            # map vae_fc1 feature to latent mean and variance
            self.vae_fc2 = nn.Linear(self.embed_dim, 2 * args.latent_size)

            # map latent variable to new feature used to predict bag of word
            if self.with_latent_bow_logits:
                if self.extend_latent_with_cls:
                    self.vae_fc3 = nn.Linear(self.embed_dim + args.latent_size, self.embed_dim)
                else:
                    self.vae_fc3 = nn.Linear(args.latent_size, self.embed_dim)
                    warnings.warn(message='using this setting may cause the model hard to be trained !')
                self.bow_fc_latent = bow_embed_tokens_latent if not self.share_all_embeddings else None
            else:
                self.vae_fc3 = None
                self.bow_fc_latent = None
                warnings.warn(message='using this setting do not utilize information the of latent variables!')

        # for mask language model loss
        if self.with_mask_lm_logits:
            self.mask_lm_fc1 = nn.Linear(self.embed_dim, self.embed_dim)
            self.mask_lm_fc2 = mask_embed_tokens if not self.share_all_embeddings else None

        self.apply(init_bert_params)

    def forward_embedding(self, src_tokens, sentence_positions=None, role_positions=None):
        # this function is for absolute position embedding, which considering token/turn/role embedding
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)

        # token absolute position embedding
        if self.embed_ap_token is not None:
            x = embed + self.embed_ap_token.forward(src_tokens)

        # turn absolute position embedding
        if self.embed_ap_sentence is not None and sentence_positions is not None:
            # assert sentence_positions is not None, \
            #     '`sentence_positions` should not be None if `self.embed_positions_sentence` is not None'
            x = x + self.embed_ap_sentence(src_tokens, positions=sentence_positions)

        # role absolute position embedding
        if self.embed_ap_role is not None and role_positions is not None:
            # assert role_positions is not None, \
            #     '`role_positions` should not be None if `self.embed_positions_role` is not None'
            x = x + self.embed_ap_role(src_tokens, positions=role_positions)

        # layer norm of embeddings
        # x = self.emb_layer_norm(x)
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x, embed

    def forward(
            self, src_tokens, src_lengths=None,
            sentence_positions=None, role_positions=None,
            relative_position_token=None, relative_position_sentence=None,
            masked_tokens=None, **kwargs
    ):
        """
        Args:
            src_tokens: tokens in the source language of shape `(batch, src_len)`
            src_lengths: lengths of each source sentence of shape `(batch)`
            sentence_positions: sentence absolute positions `(batch, src_len)`
            role_positions: role absolute positions `(batch, src_len)`
            relative_position_token: token relative positions `(batch, src_len, src_len)`
            relative_position_sentence: sentence relative positions `(batch, src_len, src_len)`
            masked_tokens: masked positions `(batch, max_masked_len)`
        """

        x, _ = self.forward_embedding(src_tokens, sentence_positions, role_positions)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layer_drop):
                # relative position embedding (PE) here
                if self.embed_rp_sentence is None and self.embed_rp_token is None:
                    x = layer(x, encoder_padding_mask=encoder_padding_mask)
                else:
                    x = layer(
                        x, encoder_padding_mask=encoder_padding_mask,
                        relative_position_token=relative_position_token,
                        relative_position_sentence=relative_position_sentence)

        cls_feature, cls_bow_logits, z, kl, masked_logits, latent_bow_logits = None, None, None, None, None, None

        if self.with_cls_bow_logits:
            cls_feature = self.map_cls_to_feature(x)
            cls_bow_logits = self.forward_cls_bow_logits(cls_feature)

        if self.generate_latent_variable:
            if cls_feature is None:
                cls_feature = self.map_cls_to_feature(x)
            # sample latent variable mean and variance
            mu, log_var = self.sample(cls_feature)
            # connect to latent space
            z = self.connect(mu, log_var, self.deterministic)
            # calculate k-l
            if not self.disable_kl_loss:
                kl = self.kl_divergence(mu, log_var)
            if self.with_latent_bow_logits:
                latent_feature = self.map_latent_to_feature(z, cls_feature)
                latent_bow_logits = self.forward_latent_bow_logits(latent_feature)

        if self.with_mask_lm_logits and masked_tokens is not None:
            masked_feature = self.map_masked_to_feature(x, masked_tokens)
            masked_logits = self.forward_masked_lm_logits(masked_feature)

        return {
            'encoder_out': x,                                   # `seq_len, batch_size, embedding_dim`
            'encoder_padding_mask': encoder_padding_mask,       # `batch_size, seq_len`
            'cls_bow_logits': cls_bow_logits,                   # `batch_size, dictionary_dim`
            'z': z,                                             # `batch_size, latent_dim`
            'kl': kl,                                           # `batch_size`
            'latent_bow_logits': latent_bow_logits,             # `batch_size, dictionary_dim`
            'masked_logits': masked_logits,                     # `batch_size, max_masked_len, dictionary_dim`
        }

    @staticmethod
    def re_parameterize(mu, log_var):
        # import torch
        # log_var = torch.zeros(4)
        # sampled latent variable using re-parameterize trick
        std = log_var.mul(.5).exp()
        eps = torch.zeros_like(std).normal_()
        return mu + torch.mul(eps, std)

    @staticmethod
    def kl_divergence(mu, log_var):
        # calculate k-l loss need to be optimized
        # cited papers:
        # 1. generating sentences from a continuous space
        # 2. improving variational inference with inverse autoregressive flow
        # 3. improving variational encoder-decoders in dialogue generation
        # 4. learning discourse-level diversity for neural dialog models using conditional variational auto-encoders
        kl_loss = .5 * (mu.pow(2.0) + log_var.exp() - log_var - 1.0)
        return kl_loss.sum(dim=1)

    # connect [CLS] feature to a hidden space
    def sample(self, feature):
        # this function return mean and log variance of latent variable
        # convert [CLS] feature to mean and log variance
        mu, log_var = self.vae_fc2(feature).chunk(2, -1)
        return mu, log_var

    def connect(self, mu, log_var, deterministic):
        # during inference, sampled process is deterministic if variance equals to zero
        if deterministic:
            log_var.fill_(.0)
        # re-parameterization
        z = self.re_parameterize(mu, log_var)
        return z

    def map_cls_to_feature(self, x):
        """
        Args:
            x: input hidden tensor of shape `seq_len, batch, embedding_dim`
        """
        # map [CLS] hidden to feature to generate mean and log variance of latent variable
        hidden = x[0]
        # feature of hidden of [CLS] token, as well as feature of `posterior distribution`
        feature = torch.tanh(self.vae_fc1(hidden))
        return feature

    def forward_cls_bow_logits(self, feature):
        # this function return bag-of-word distribution need to be optimized
        # and the feature is generated by encoder [CLS]
        assert self.with_cls_bow_logits, '`with_encoder_bow_logits` parameter should be set to true!'
        # bag of words distribution
        if self.share_all_embeddings:
            # if share_all_embeddings, use Embedding layer's parameter to compute bag of word distribution
            bow_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.bow_fc_enc is not None, '`self.bow_fc_enc` should not be None!'
            bow_logits = self.bow_fc_enc(feature)
        return bow_logits

    def map_latent_to_feature(self, z, cls_feature):
        """
        Args:
            z: latent variable tensor of shape `batch, latent_dim`
            cls_feature: cls feature of shape `batch, embedding_dim`
        """
        if self.extend_latent_with_cls:
            hidden = torch.cat([z, cls_feature], dim=1)
        else:
            hidden = z
        feature = torch.tanh(self.vae_fc3(hidden))
        return feature

    def forward_latent_bow_logits(self, feature):
        # this function return bag-of-word distribution need to be optimized
        # and the feature is generated by latent variable, possible with encoder feature
        assert self.with_latent_bow_logits, '`with_encoder_latent_logits` parameter should be set to true!'
        # bag of words distribution
        if self.share_all_embeddings:
            # if share_all_embeddings, use Embedding layer's parameter to compute bag of word distribution
            bow_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.bow_fc_latent is not None, '`self.bow_fc_latent` should not be None!'
            bow_logits = self.bow_fc_latent(feature)
        return bow_logits

    def map_masked_to_feature(self, x, masked_position):
        """
        Args:
            x: input hidden tensor of shape `seq_len, batch, embedding_dim`
            masked_position: masked position tensor of shape `batch_size, max_masked_len`
        """
        # map all masked tokens to feature to predict masked tokens
        _, _, embed_dim = x.shape
        _x = x.permute(1, 0, 2).contiguous()
        hidden = torch.gather(_x, 1, masked_position.unsqueeze(-1).repeat(1, 1, embed_dim).long())
        # feature of hidden of masked tokens
        feature = torch.tanh(self.mask_lm_fc1(hidden))
        return feature

    def forward_masked_lm_logits(self, feature):
        # this function return masked tokens distribution need to be optimized
        # and the feature is generated by masked token hidden
        assert self.with_mask_lm_logits, '`with_mask_lm_logits` parameter should be set to true!'
        if self.share_all_embeddings:
            # if share_all_embeddings, use Embedding layer's parameter to compute predicted distribution
            masked_logits = F.linear(feature, weight=self.embed_tokens.weight)
        else:
            assert self.mask_lm_fc2 is not None, '`self.mask_lm_fc2` should not be None'
            masked_logits = self.mask_lm_fc2(feature)
        return masked_logits

    # training
    def reorder_encoder_out(self, encoder_out, new_order):          # -> inference
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['cls_bow_logits'] is not None:
            encoder_out['cls_bow_logits'] = encoder_out['cls_bow_logits'].index_select(0, new_order)
        if encoder_out['z'] is not None:
            encoder_out['z'] = encoder_out['z'].index_select(0, new_order)
        if encoder_out['kl'] is not None:
            encoder_out['kl'] = encoder_out['kl'].index_select(0, new_order)
        if encoder_out['latent_bow_logits'] is not None:
            encoder_out['latent_bow_logits'] = encoder_out['latent_bow_logits'].index_select(0, new_order)
        if encoder_out['masked_logits'] is not None:
            encoder_out['masked_logits'] = encoder_out['masked_logits'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        if self.embed_ap_token is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_ap_token.max_positions())


# Transformer Decoder Layer (similar to encoder except for encoder-decoder attention layer)
# original paper method: dropout -> add residual -> layer_norm
# multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
# encoder-decoder-attention sublayer: x = layer_norm(x + dropout(encoder_attn(x))
# position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
#
# tensor2tensor method: layer_norm -> dropout -> add residual
# multi-head self-attention sublayer: x = x + dropout(self_attn(layer_norm(x))))
# encoder-decoder-attention sublayer: x = x + dropout(encoder_attn(layer_norm(x)))
# position-wise feed-forward sublayer: x = x + dropout(fc2(dropout(relu(fc1(layer_norm(x))))))


class NgramTransformerDecoderLayer(nn.Module):
    def __init__(
            self, ngram, encoder_embed_dim, decoder_embed_dim, decoder_ffn_embed_dim, decoder_attention_heads,
            dropout, attention_dropout, activation_dropout, activation_fn, with_encoder_decoder_attn
    ):
        super().__init__()

        self.with_encoder_decoder_attn = with_encoder_decoder_attn
        self.embed_dim = decoder_embed_dim
        self.ngram_self_attn = NgramMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=decoder_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            ngram=ngram
        )

        self.dropout = dropout
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.activation_dropout = activation_dropout

        self.ngram = ngram
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if self.with_encoder_decoder_attn:
            # encoder-decoder layer
            self.encoder_attn = MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=decoder_attention_heads,
                kdim=encoder_embed_dim,
                vdim=encoder_embed_dim,
                dropout=attention_dropout,
                self_attention=False,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        else:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None

        self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = nn.Linear(decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = False

    def forward(
            self, x, encoder_out=None, encoder_mask=None, incremental_state=None, prev_self_attn_state=None,
            prev_attn_state=None, self_attn_mask=None, ngram_mask_matrix=None,
            i_buckets_main_stream=None, i_bucket_relative_stream=None, real_positions=None,
            latent_context=None
    ):
        # multi-head self-attention sublayer: x = layer_norm(x + dropout(self_attn(x)))
        residual = x
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        # unify ngram_self_attn module to MultiheadAttentionRPE module
        x, attn = self.ngram_self_attn(
            query=x,
            incremental_state=incremental_state,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions,
            latent_context=latent_context,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # if training-mode is vae, encoder-decoder attention should not be used.
        # decoder should only depend on latent variable
        if self.with_encoder_decoder_attn:
            # encoder-decoder-attention sublayer: x = layer_norm(x + dropout(encoder_attn(x)))
            residual = x
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.encoder_attn_layer_norm(x)

        # position-wise feed-forward sublayer: x = layer_norm(x + dropout(fc2(dropout(relu(fc1(x))))))
        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class NgramTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        # self.training_mode = args.training_mode
        self.with_encoder_decoder_attn = args.with_encoder_decoder_attn
        self.ngram = args.ngram
        self.num_buckets = args.num_buckets
        self.relative_max_distance = args.relative_max_distance
        self.max_target_positions = args.max_target_positions
        self.max_target_positions_token = args.max_target_positions_token
        self.use_latent_variable = True if args.use_latent_variable else False
        self.dropout = args.dropout

        self.embed_dim = args.decoder_embed_dim
        assert embed_tokens.embedding_dim == self.embed_dim, \
            '`encoder_embed_dim` parameter in global args must equal to the one in self-attention'
        self.decoder_attention_heads = args.decoder_attention_heads
        self.head_dim = self.embed_dim // self.decoder_attention_heads

        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        # self.embed_scale = None
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(args.decoder_embed_dim)

        self.embed_ap_token = LearnedPositionalEmbeddingNew(
            self.max_target_positions_token + 2 + self.padding_idx, self.embed_dim, self.padding_idx,
        )

        self.ngram_input_embed = Embedding(self.ngram, self.embed_dim, None)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NgramTransformerDecoderLayer(
                args.ngram,
                args.encoder_embed_dim,
                self.embed_dim,
                args.decoder_ffn_embed_dim,
                args.decoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
                self.with_encoder_decoder_attn
            )
            for _ in range(args.decoder_layers)
        ])

        self.emb_layer_norm = LayerNorm(self.embed_dim)

        # incorporate latent variable in VAE to decoder, time 2 because both key and value are generated
        # if not self.with_encoder_decoder_attn:
        #     self.vae_transform = nn.Linear(
        #         in_features=args.latent_size, out_features=2 * self.embed_dim)
        # else:
        #     self.vae_transform = None

        if self.use_latent_variable:
            self.vae_transform = nn.Linear(
                in_features=args.latent_size, out_features=2 * self.embed_dim)
        else:
            self.vae_transform = None
        self.apply(init_bert_params)

    def forward(
            self, prev_output_tokens, encoder_out=None,
            incremental_state=None, vae_hidden=None, **kwargs
    ):
        x_list, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **kwargs)
        x_predicted = x_list[1:]
        x_predicted = [self.output_layer(x) for x in x_predicted]
        if incremental_state is not None:
            x_predicted = x_predicted[0]
            for k in extra:
                if extra[k] is not None:
                    extra[k] = extra[k][0]
        return x_predicted, extra

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def cal_pretrain_relative_positions(self, real_positions):
        main_stream_relative_positions = real_positions.unsqueeze(1)
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main
        real_positions_shift_predicting_stream = real_positions - 1
        predicting_stream_relative_positions = torch.cat((
            real_positions_shift_predicting_stream, real_positions), dim=-1).unsqueeze(1)
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(
            1, real_positions.size(-1), 1)
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(
            predicting_stream_relative_positions, bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)

        if not hasattr(self, '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None or \
                self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)

        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(
            batch_size, 1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    def transform_latent_context(self, latent_context):
        # transform to latent key and value
        # latent_context -> [batch_size, latent_dim]
        latent_context = self.vae_transform(latent_context)
        # latent_context -> [batch_size, 2 * embedding_dim]
        # -> [batch_size * num_heads, 1, 2 * head_dim]
        # [bsz * self.num_heads, 1, self.head_dim]
        # -> bsz * self.num_heads, 1, self.head_dim
        latent_context = latent_context.view(-1, 1, 2 * self.head_dim)
        latent_context = torch.split(latent_context, self.head_dim, dim=-1)
        return latent_context

    def extract_features(
            self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):

        main_stream_pos_embed, real_positions = self.embed_ap_token(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_ap_token is not None else None

        if incremental_state is not None:
            i_buckets_main_stream, i_bucket_relative_stream = None, None
        else:
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_finetune_relative_positions(real_positions)

        # calculate predicting stream position embedding
        predicting_stream_pos_embed = self.embed_ap_token._forward(real_positions + 1)

        if incremental_state is not None:
            # understand that, in the incremental decoding, only the last output is need for decoding
            prev_output_tokens = prev_output_tokens[:, -1:]
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        x = self.embed_tokens(prev_output_tokens)
        if self.embed_scale is not None:
            x *= self.embed_scale

        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed

        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight
        # ngram embedding

        if incremental_state is not None:
            B = x.size(1)
            ngram_masks = [
                (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
        else:
            ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None

        x = torch.cat([x] + ngram_masks, 0)

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # transform latent variable to first key and value for main-stream used in ProphetNet
        #
        # inject latent variable to decoder, i.e., self-attention in decoder
        # new_attn = MutliHead(q, [k, k0], [v, v0])
        #
        # key0, value0   key1, value1   key2, value2   key3, value3
        #                   token1         token2        token3
        #
        # the predicting stream can use latent variable implicitly by the hidden state of main-stream
        # parameters of the latent key and value is shared across all decoder layers

        if not self.use_latent_variable:
            latent_context = None
        else:
            latent_context = self.transform_latent_context(encoder_out['z'])

        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions,
                # 2-element tensor tuple which has a shape [batch_size * num_heads, 1, head_dim]
                latent_context=latent_context,
            )
            inner_states.append(x)

        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        if attn is not None:
            attn_list = attn.transpose(0, 1).chunk(1 + self.ngram, 1)
        else:
            attn_list = None

        return x_list, {'attn': attn_list}

    def get_normalized_probs(self, net_output, log_probs, sample):
        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def output_layer(self, features, **kwargs):
        return F.linear(features, self.embed_tokens.weight)

    def max_positions(self):
        if self.embed_ap_token is None:
            return self.max_target_positions_token
        return min(self.max_target_positions, self.embed_ap_token.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or \
                self._future_mask is None or \
                self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_ngram_future_mask') or \
                self._ngram_future_mask is None or \
                self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(
                self.max_target_positions_token, self.ngram).type(tensor.dtype).to(tensor.device)
        ngram_future_mask = torch.cat([
            self._ngram_future_mask[:, :dim, :dim],
            self._ngram_future_mask[:, :dim, self.max_target_positions_token: self.max_target_positions_token + dim]], 2)
        return ngram_future_mask


def base_architecture(args):
    # default parameters in fairseq cannot be modified twice in this function and
    # they will exist by default when initializing
    # if the default parameters are need to be modified not in command line,
    # please assign it in this function directly, like below:

    # encoder architecture
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layer_drop = getattr(args, 'encoder_layer_drop', 0.0)

    # encoder position embedding
    args.with_encoder_ape_token = getattr(args, 'with_encoder_ape_token', True)

    # default is False
    args.with_encoder_ape_sentence = getattr(args, 'with_encoder_ape_sentence', False)
    args.with_encoder_ape_role = getattr(args, 'with_encoder_ape_role', False)

    # encoder absolute position embedding
    # args.max_source_positions_token = getattr(args, 'max_source_positions_token', args.max_source_positions)
    args.max_source_positions_token = getattr(args, 'max_source_positions_token', 512)
    # suppose that total rounds of dialogue are no more than 32
    args.max_source_positions_sentence = getattr(args, 'max_source_positions_sentence', 32)
    # 1 for the response, 2 for the other character, 3 for the possible background knowledge
    args.max_source_positions_role = getattr(args, 'max_source_positions_role', 4)

    # encoder relative position embedding
    args.num_buckets_source_token = getattr(args, 'num_buckets_source_token', 32)
    args.max_distance_source_token = getattr(args, 'max_distance_source_token', 128)
    args.bidirectional_source_token = getattr(args, 'bidirectional_source_token', True)

    args.num_buckets_source_sentence = getattr(args, 'num_buckets_source_sentence', 4)
    args.max_distance_source_sentence = getattr(args, 'max_distance_source_sentence', 16)
    args.bidirectional_source_sentence = getattr(args, 'bidirectional_source_sentence', True)

    # default is False
    args.with_encoder_rpe_token = getattr(args, 'with_encoder_rpe_token', False)
    args.with_encoder_rpe_sentence = getattr(args, 'with_encoder_rpe_sentence', False)

    # decoder
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)

    # decoder position embedding
    args.max_target_positions_token = getattr(args, 'max_target_positions_token', 512)

    # decoder relative position embedding
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    # common
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)

    args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
    args.deterministic = getattr(args, 'deterministic', False)


# ----------------------------------------------------------------------------------------------------------------------
# released pre-trained models

# masked language loss + response generation loss
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq_pretrain')
def transformer_seq2seq_pretrain(args):
    # this parameter is the key difference between seq2seq-based and vae-based models
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', False)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', False)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)


# masked language loss + response generation loss + bag-of-word loss + K-L loss
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_standard_pretrain')
def transformer_vae_standard_pretrain(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', True)
    args.latent_size = getattr(args, 'latent_size', 32)


# masked language loss + response generation loss + bag-of-word loss + K-L loss
@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_large_pretrain')
def transformer_vae_large_pretrain(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', True)
    args.latent_size = getattr(args, 'latent_size', 64)


# ----------------------------------------------------------------------------------------------------------------------
# released fine-tuned models

@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq')
def transformer_seq2seq(args):
    # this parameter is the key difference between seq2seq-based and vae-based models
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', False)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', False)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_standard')
def transformer_vae_standard(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)

    # standard VAE transformer has latent size equal to 32
    args.latent_size = getattr(args, 'latent_size', 32)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_large')
def transformer_vae_large(args):
    args.generate_latent_variable = getattr(args, 'generate_latent_variable', True)
    args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
    args.use_latent_variable = getattr(args, 'use_latent_variable', True)

    args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
    args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
    args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
    args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)

    # large VAE transformer has latent size equal to 64
    args.latent_size = getattr(args, 'latent_size', 64)


# ----------------------------------------------------------------------------------------------------------------------
# other pre-trained models

# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq_base')
# def transformer_seq2seq_base(args):
#     args.generate_latent_variable = False
#     args.with_encoder_decoder_attn = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', True)
#
#
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq_base_no_mask')
# def transformer_seq2seq_base_no_mask(args):
#     args.generate_latent_variable = getattr(args, 'generate_latent_variable', False)
#     args.with_encoder_decoder_attn = getattr(args, 'with_encoder_decoder_attn', True)
#
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', False)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', True)
#
#
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_seq2seq_cls')
# def transformer_seq2seq(args):
#     args.generate_latent_variable = False
#     args.with_encoder_decoder_attn = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', True)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', True)
#
#
# # this is the first version we implemented
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae')
# def transformer_vae(args):
#     args.generate_latent_variable = True
#     args.with_encoder_decoder_attn = True
#     args.use_latent_variable = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', True)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', False)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     # variational auto-encoder
#     args.latent_size = getattr(args, 'latent_size', 32)
#     args.deterministic = getattr(args, 'deterministic', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
#
#
# # this is the version we think is best
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_megatron')
# def transformer_vae_megatron(args):
#     args.generate_latent_variable = True
#     args.with_encoder_decoder_attn = True
#     args.use_latent_variable = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     # variational auto-encoder
#     args.latent_size = getattr(args, 'latent_size', 32)
#     args.deterministic = getattr(args, 'deterministic', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
#
#
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_kojiro')
# def transformer_vae_kojiro(args):
#     args.generate_latent_variable = True
#     args.with_encoder_decoder_attn = False
#     args.use_latent_variable = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', True)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     # variational auto-encoder
#     args.latent_size = getattr(args, 'latent_size', 32)
#     args.deterministic = getattr(args, 'deterministic', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
#
#
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_cobra')
# def transformer_vae_cobra(args):
#     args.generate_latent_variable = True
#     args.with_encoder_decoder_attn = False
#     args.use_latent_variable = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', False)
#     # variational auto-encoder
#     args.latent_size = getattr(args, 'latent_size', 32)
#     args.deterministic = getattr(args, 'deterministic', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
#
#
# @register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_vae_thanos')
# def transformer_vae_thanos(args):
#     args.generate_latent_variable = True
#     args.with_encoder_decoder_attn = False
#     args.use_latent_variable = True
#     args.with_mask_lm_logits = getattr(args, 'with_mask_lm_logits', True)
#     args.with_cls_bow_logits = getattr(args, 'with_cls_bow_logits', False)
#     args.with_latent_bow_logits = getattr(args, 'with_latent_bow_logits', True)
#     args.extend_latent_with_cls = getattr(args, 'extend_latent_with_cls', True)
#     # variational auto-encoder
#     args.latent_size = getattr(args, 'latent_size', 32)
#     args.deterministic = getattr(args, 'deterministic', False)
#     args.disable_kl_loss = getattr(args, 'disable_kl_loss', False)
#
#
# # some setting is for ablation experiment
# # most ablation experiments are only implemented in the fine-tune stage
