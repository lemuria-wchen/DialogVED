# this script is for Transformer Encoder & Decoder self-attention with relative position embedding
# it support both token-level and sentence-level relative position
# our implementation refers to T5
# see more detail in < https://github.com/google-research/text-to-text-transfer-transformer >
# TODO: in the released version, we will merge some modules to reduce this part of code
import math
import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from fairseq import utils
from fairseq.modules import MultiheadAttention


def _relative_position_bucket(relative_position, num_buckets, max_distance, bidirectional):
    n = -relative_position
    ret = 0
    if bidirectional:
        num_buckets = num_buckets // 2
        ret = ret + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = torch.lt(n, max_exact)

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
            num_buckets - max_exact)
    val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
    val_if_large = val_if_large.int()

    ret = ret + torch.where(is_small, n.int(), val_if_large)
    return ret


class RelativePositionBias(nn.Module):
    def __init__(self, embed_dim, n_heads, num_buckets, max_distance, bidirectional):
        super(RelativePositionBias, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        self.relative_attention_bias = nn.Linear(embed_dim, num_buckets * n_heads)

        if self.num_buckets % 4 != 0 or self.max_distance % self.num_buckets != 0:
            warnings.warn(message='`num_buckets` is suggest to be a multiple of 4, and '
                                  '`max_distance` is suggest to be a multiple of `num_buckets`')

    def compute_bias(self, query, relative_position):
        """
        :param query: input query (seq_len, batch_size, embed_dim)
        :param relative_position: position matrix (batch_size, src_len, tgt_len)
        :return: attention bias: (batch_size * num_heads, src_len, tgt_len)
        """
        # assert relative_position is not None, '`relative_position` must be provided'
        # if relative_position is None:
        # assert size is not None, '`size` should not be None if `relative_position` is None'
        # src_len, tgt_len = size
        # relative_position = _infer_relative_position_token(src_len, tgt_len)

        tgt_len, batch_size, _ = query.shape
        _, src_len, tgt_len = relative_position.shape

        # relative_position (batch_size, src_len, tgt_len) -> bucket (batch_size, tgt_len, src_len)
        bucket = _relative_position_bucket(
            relative_position, self.num_buckets, self.max_distance, self.bidirectional)

        # bucket (tgt_len, src_len) -> bucket (num_buckets * n_heads, tgt_len, src_len)
        # bucket (num_buckets * n_heads, tgt_len, src_len) -> (num_buckets * n_heads * tgt_len, src_len)
        _bucket = bucket.transpose(1, 0).squeeze(0).repeat(self.n_heads, 1, 1).view(-1, src_len).long()

        # query (tgt_len, batch_size, embed_dim) -> values (tgt_len, batch_size, num_buckets * n_heads)
        attn_bias = self.relative_attention_bias(query)
        # values -> (tgt_len, batch_size, num_buckets, n_heads)
        attn_bias = attn_bias.view(tgt_len, batch_size, self.num_buckets, self.n_heads)
        # values -> (batch_size, n_heads, tgt_len, num_buckets)
        attn_bias = attn_bias.permute([1, 3, 0, 2])
        # values -> (batch_size * n_heads * tgt_len, num_buckets)
        attn_bias = attn_bias.contiguous().view(-1, self.num_buckets)

        bucket_attn_bias = torch.gather(attn_bias, dim=1, index=_bucket)
        bucket_attn_bias = bucket_attn_bias.view(-1, tgt_len, src_len)
        return bucket_attn_bias, bucket

    def forward(self, query, relative_position):
        return self.compute_bias(query, relative_position)


class MultiheadAttentionRPE(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, self_attention=False, encoder_decoder_attention=False,
                 embed_relative_positions_token=None, embed_relative_positions_sentence=None):

        super().__init__(
            embed_dim, num_heads, kdim, vdim, dropout, bias,
            add_bias_kv, add_zero_attn, self_attention, encoder_decoder_attention)

        self.embed_token = embed_relative_positions_token
        self.embed_sentence = embed_relative_positions_sentence

    def forward(
            self,
            query, key, value,
            key_padding_mask=None,
            incremental_state=None,
            need_weights=True,
            static_kv=False,
            attn_mask=None,
            before_softmax=False,
            need_head_weights=False,
            relative_position_token=None,
            relative_position_sentence=None,
    ):
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        token_attn_bias, sentence_attn_bias = torch.zeros_like(attn_weights), torch.zeros_like(attn_weights)

        """add attention relative weight here, aka., relative position embedding"""
        # if self.embed_token is not None and relative_position_token is not None:
        if self.embed_token is not None and relative_position_token is not None:
            # assert relative_position_token is not None, \
            #     '`relative_position_token` should not be None, ' \
            #     'if `no_enc_relative_positions_embedding_token` parameter is set to false`'
            token_attn_bias, _ = self.embed_token(query, relative_position_token)
            attn_weights = attn_weights + token_attn_bias

        # if self.embed_sentence is not None and relative_position_sentence is not None:
        if self.embed_sentence is not None and relative_position_sentence is not None:
            # assert relative_position_sentence is not None, \
            #     '`relative_position_sentence` should not be None, ' \
            #     'if `no_enc_relative_positions_embedding_sentence` parameter is set to false`'
            # update sentence position here (updated)
            sentence_attn_bias, _ = self.embed_sentence(query, relative_position_sentence)
            attn_weights = attn_weights + sentence_attn_bias

        # if self.embed_token is not None and self.embed_sentence is not None \
        #         and relative_position_token is not None and relative_position_sentence is not None:
        if self.embed_token is not None and self.embed_sentence is not None and \
                relative_position_token is not None and relative_position_sentence is not None:
            # assert relative_position_token is not None and relative_position_sentence is not None, \
            #     '`relative_position_sentence` and `relative_position_sentence` should not be None, if' \
            #     '`no_enc_relative_positions_embedding_token` and ' \
            #     '`no_enc_relative_positions_embedding_sentence` parameter is set to false`'
            # consider the interaction of token bias and sentence bias
            attn_weights = attn_weights + torch.mul(token_attn_bias, sentence_attn_bias)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # TODO: introduce this module to @yinzi
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, attn_weights


class NgramMultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False, ngram=2, num_buckets=32, relative_max_distance=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.ngram = ngram

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and ' \
                                                             'value to be of the same size'

        self.relative_linear = nn.Linear(embed_dim, num_buckets * num_heads)
        if self.qkv_same_dim:
            self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)
            nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

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

    def main_stream_relative_logits(self, query, attn_weights, real_positions, i_bucket_main_stream):
        # input query [T,B,C]
        # input attn_weights [T*head,T,S]
        # input real_positions [B,T] or [1,1]

        T, B, _ = query.size()
        S = attn_weights.size(-1)

        if i_bucket_main_stream is not None:
            i_buckets = i_bucket_main_stream
        else:
            # [B,T,S]
            # start from 2
            relative_positions = torch.arange(2, S + 2).unsqueeze(0).unsqueeze(0).repeat(B, T, 1).to(
                real_positions.device)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B, T, 1)
            # [B,T,S]
            relative_positions = relative_positions - real_positions
            # [B,T,T]
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [B,T,C]
        query = query.transpose(0, 1)
        # [B,T,Buckets*head]
        values = self.relative_linear(query)
        # [B,T,Buckets,head]
        values = values.view(values.size(0), values.size(1), self.num_buckets, self.num_heads)
        # [B,head,Buckets,T]
        values = values.transpose(1, 3)
        # [B,head,T,Buckets]
        values = values.transpose(2, 3)
        # [B*head,T,Buckets]
        values = values.reshape(attn_weights.size(0), attn_weights.size(1), -1)

        # => [B,head*T,T] => [B*head,T,T]
        i_buckets = i_buckets.repeat(1, self.num_heads, 1).view(attn_weights.size(0), attn_weights.size(1), -1)
        # [B*head*T,Buckets]
        values = values.reshape(-1, values.size(-1))
        # [B*head*T,T]
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [B*head*T,T]
        result = torch.gather(values, dim=1, index=i_buckets)
        # [B*head,T,T]
        result = result.view(attn_weights.size(0), attn_weights.size(1), -1)

        return result

    def ngram_relative_logits(self, query, attn_weights, real_positions, i_bucket_relative_stream):
        # input query [ngram, T,B,C]
        # input attn_weights [ngram, B*head,T,S]
        # input real_positions [B,T] or [1,1]
        # input i_bucket_relative_stream [B,T, 2*T] or None

        N, T, B, _ = query.size()
        _, BH, _, S = attn_weights.size()

        if i_bucket_relative_stream is not None:
            i_buckets = i_bucket_relative_stream
        else:
            # [B,T,S]
            # assert real_positions[0][0] == S, 'memory position is 1 2 3 4 5(S-1)'
            relative_positions = torch.arange(1, S + 1).unsqueeze(0).unsqueeze(0).repeat(B, T, 1).to(
                real_positions.device)
            # print('relative_positions', relative_positions)
            # [B,T,1]
            real_positions = real_positions.unsqueeze(0).repeat(B, T, 1)
            relative_positions = relative_positions
            # [B,T,2*T] or [B,T,S]
            relative_positions = relative_positions - real_positions
            i_buckets = self._relative_positions_bucket(relative_positions, False)

        # [ngram, B, T, C]
        query = query.transpose(1, 2)
        # [ngram, B, T, bucket*head]
        values = self.relative_linear(query)
        # [ngram, B, T, bucket, head]
        values = values.view(*values.size()[:-1], self.num_buckets, self.num_heads)
        # [ngram, B, head, T, bucket]
        values = values.permute(0, 1, 4, 2, 3)
        # [ngram*B*head, T, bucket]
        values = values.reshape(N * BH, T, -1)

        # [ngram, B, head*T, S]
        i_buckets = i_buckets.unsqueeze(0).repeat(N, 1, self.num_heads, 1)

        values = values.reshape(-1, values.size(-1))
        i_buckets = i_buckets.view(-1, i_buckets.size(-1)).long()
        # [ngram*B*head*T, S]
        result = torch.gather(values, dim=1, index=i_buckets)
        # [ngram, B*head, T, S]
        result = result.view(N, BH, T, -1)

        return result

    def forward(self, query,
                incremental_state=None,
                static_kv=False,
                self_attn_mask=None,
                ngram_mask_matrix=None,
                i_buckets_main_stream=None,
                i_bucket_relative_stream=None,
                real_positions=None,
                latent_context=None,
                ):

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        # q,k,v -> [3 * seq_len, batch_size, embedding_dim]
        q, k, v = self.in_proj_qkv(query)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

        # q, k, v -> [batch_size * num_heads, 3 * seq_len, head_dim]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # h_list 3-dim list -> [seq_len, batch_size, num_heads * head_dim]
        h_list = query.chunk(1 + self.ngram, dim=0)

        # q_list 3-dim list -> [batch_size * num_heads, seq_len, head_dim]
        q_list = q.chunk(1 + self.ngram, dim=1)
        k_list = k.chunk(1 + self.ngram, dim=1)
        v_list = v.chunk(1 + self.ngram, dim=1)

        # q/k/v main -> [batch_size * num_heads, seq_len, head_dim]
        # h_main -> [seq_len, batch_size, num_heads * head_dim]
        h_main, h_predict_list = h_list[0], h_list[1:]
        q_main, q_predict_list = q_list[0], q_list[1:]
        k_main, k_predict_list = k_list[0], k_list[1:]
        v_main, v_predict_list = v_list[0], v_list[1:]

        # # incremental_state 在的training和validation的阶段是None，在测试的第一步是 {}
        if not incremental_state and latent_context is not None:
            context_k, context_v = latent_context
            # context_k = torch.rand(bsz * self.num_heads, 1, self.head_dim).to('cuda').half()
            # context_v = torch.rand(bsz * self.num_heads, 1, self.head_dim).to('cuda').half()
            k_main = torch.cat([context_k, k_main], dim=1)
            v_main = torch.cat([context_v, v_main], dim=1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    assert False, 'static_kv not supprt in ngram decoder'
                    k = prev_key
                else:
                    k_main = torch.cat((prev_key, k_main), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v_main = torch.cat((prev_value, v_main), dim=1)
            saved_state['prev_key'] = k_main.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v_main.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        real_tgt_len = tgt_len // (1 + self.ngram)

        attn_weights_main = torch.bmm(q_main, k_main.transpose(1, 2))

        if not incremental_state and latent_context is not None:
            # we do not calculate new relative position embedding of the latent variable's key and value
            main_relative_logits = self.main_stream_relative_logits(
                h_main, attn_weights_main[:, :, 1:], real_positions, i_buckets_main_stream)
            attn_weights_main[:, :, 1:] = attn_weights_main[:, :, 1:] + main_relative_logits
        else:
            main_relative_logits = self.main_stream_relative_logits(
                h_main, attn_weights_main, real_positions, i_buckets_main_stream)
            attn_weights_main = attn_weights_main + main_relative_logits

        if self_attn_mask is not None:
            if not incremental_state and latent_context is not None:
                self_attn_mask = self_attn_mask.unsqueeze(0)
                attn_weights_main[:, :, 1:] = attn_weights_main[:, :, 1:] + self_attn_mask
            else:
                self_attn_mask = self_attn_mask.unsqueeze(0)
                attn_weights_main = attn_weights_main + self_attn_mask

        attn_weights_main = utils.softmax(
            attn_weights_main, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_main)
        attn_weights_main = F.dropout(attn_weights_main, p=self.dropout, training=self.training)

        attn_main = torch.bmm(attn_weights_main, v_main)
        attn_main = attn_main.transpose(0, 1).contiguous().view(1, real_tgt_len, bsz, embed_dim)
        attn_main = self.out_proj(attn_main)

        # [ngram, B*head, T, c]
        q_ngram = torch.cat(q_predict_list, 0).view(self.ngram, -1, real_tgt_len, self.head_dim)
        # [ngram, B*head, 2*T, c]

        # 这里我们不需要将 k_main 中的加入 k 和 k_p 拼起来
        if not incremental_state and latent_context is not None:
            k_ngram = torch.cat([torch.cat([k_main[:, 1:, ], k_p], 1).unsqueeze(0) for k_p in k_predict_list], 0)
        else:
            k_ngram = torch.cat([torch.cat([k_main, k_p], 1).unsqueeze(0) for k_p in k_predict_list], 0)

        # below code slower than above for loop
        # k_ngram = torch.cat([k_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(k_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, T, B, C]
        h_ngram = torch.cat(h_predict_list, 0).view(self.ngram, real_tgt_len, bsz, embed_dim)

        # [ngram, B*head, 2*T, c]
        if not incremental_state and latent_context is not None:
            v_ngram = torch.cat([torch.cat([v_main[:, 1:, :], v_p], 1).unsqueeze(0) for v_p in v_predict_list], 0)
        else:
            v_ngram = torch.cat([torch.cat([v_main, v_p], 1).unsqueeze(0) for v_p in v_predict_list], 0)
        # below code slower than above for loop
        # v_ngram = torch.cat([v_main.unsqueeze(0).repeat(self.ngram, 1, 1, 1) , torch.cat(v_predict_list).view(self.ngram, -1, real_tgt_len, self.head_dim)], 2)

        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = torch.einsum('nbtc,nbsc->nbts', (q_ngram, k_ngram))

        # [ngram, B*head, T, S]
        predict_relative_logits = self.ngram_relative_logits(h_ngram, attn_weights_ngram, real_positions,
                                                             i_bucket_relative_stream)
        # [ngram, B*head, T, 2*T]
        attn_weights_ngram = attn_weights_ngram + predict_relative_logits

        if ngram_mask_matrix is not None:
            ngram_mask_matrix = ngram_mask_matrix.unsqueeze(1)
            attn_weights_ngram = attn_weights_ngram + ngram_mask_matrix

        attn_weights_ngram = utils.softmax(
            attn_weights_ngram, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights_ngram)
        attn_weights_ngram = F.dropout(attn_weights_ngram, p=self.dropout, training=self.training)

        # [ngram, B*head, T, c]
        attn_ngram = torch.einsum('nbts,nbsc->nbtc', (attn_weights_ngram, v_ngram))
        # [ngram, T, B, C]
        attn_ngram = attn_ngram.transpose(1, 2).contiguous().view(self.ngram, real_tgt_len, bsz, embed_dim)
        attn_ngram = self.out_proj(attn_ngram)

        attn_result = []
        attn_result.append(attn_main)
        attn_result.append(attn_ngram)

        # [1+ngram*T, B, C]
        attn = torch.cat(attn_result, 0).view(-1, bsz, embed_dim)
        return attn, None

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        if self.qkv_same_dim:
            return self._in_proj(query, end=self.embed_dim)
        else:
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        if self.qkv_same_dim:
            return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)
        else:
            weight = self.k_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[self.embed_dim:2 * self.embed_dim]
            return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        if self.qkv_same_dim:
            return self._in_proj(value, start=2 * self.embed_dim)
        else:
            weight = self.v_proj_weight
            bias = self.in_proj_bias
            if bias is not None:
                bias = bias[2 * self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


def ngram_attention_bias(length, num_skip):
    bias_result = []
    for n_skip in range(num_skip):
        bias_n_skip = []
        for i in range(length):
            bias_this = [float('-inf')] * (2 * length)
            bias_this[length + i] = 0
            first_k = i - n_skip
            first_k = first_k if first_k > 0 else 0
            for j in range(first_k + 1):
                bias_this[j] = 0
            bias_n_skip.append(bias_this)
        bias_result.append(bias_n_skip)
    return torch.from_numpy(np.array(bias_result, dtype=np.float32))


class LearnedPositionalEmbeddingNew(nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: int,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.onnx_trace = False

    def forward(self, _input, incremental_state=None, positions=None):
        assert (positions is None) or (
                self.padding_idx is None
        ), "If positions is pre-computed then padding_idx should not be set."

        if positions is None:
            if incremental_state is not None:
                positions = _input.data.new(1, 1).fill_(int(self.padding_idx + _input.size(1)))
            else:
                positions = utils.make_positions(
                    _input.data, self.padding_idx, onnx_trace=self.onnx_trace,
                )
            real_positions = positions
        else:
            real_positions = positions
        return super().forward(positions), real_positions

    def max_positions(self):
        if self.padding_idx is not None:
            return self.num_embeddings - self.padding_idx - 1
        else:
            return self.num_embeddings

    def _forward(self, positions):
        return super().forward(positions)


"""
some test:

def _relative_position_bucket_test(src_len=7, tgt_len=5):
    # attention weight should have same dimension (batch_size * num_heads, src_len, tgt_len)
    position = torch.arange(src_len).unsqueeze(0) - torch.arange(tgt_len).unsqueeze(1)
    print(position)
    print(_relative_position_bucket(position, num_buckets=32, max_distance=128, bidirectional=False))


def _relative_position_bias_test():
    rp = RelativePositionBias(
        embed_dim=64, num_buckets=32, max_distance=128, n_heads=2, bidirectional=True
    )
    query = torch.rand(7, 4, 64)
    input_ids = torch.randint(0, 64, (7, 4))
    bias, bucket = rp(query, _infer_relative_position_token(input_ids, 1))
    print(bias.shape)
    print(bucket.shape)
    print(bucket)


def _multi_head_attention_rpe_test():
    embed_token = RelativePositionBias(
        embed_dim=1024, num_buckets=32, max_distance=128,
        n_heads=8, bidirectional=True)
    embed_turn = RelativePositionBias(
        embed_dim=1024, num_buckets=8, max_distance=32,
        n_heads=8, bidirectional=True)
    self_attn = MultiheadAttentionRPE(
        1024, 8,
        embed_relative_positions_token=embed_token,
        embed_relative_positions_turn=embed_turn,
    )
    x = torch.rand(10, 6, 1024)
    input_ids = torch.randint(0, 100, (10, 6))
    print(x.shape)
    x, _ = self_attn.forward(
        x, x, x,
        relative_position_token=_infer_relative_position_token(input_ids, 1),
        relative_position_turn=_infer_relative_position_turn(input_ids, 5),
    )
    print(x.shape)


# def test():
#     input_ids = torch.tensor([
#         [4, 10, 9, 8, 5, 22, 21, 20, 19, 5, 31, 30, 29, 5, 1],
#         [4, 10, 9, 5, 23, 22, 21, 20, 19, 18, 16, 5, 1, 1, 1],
#     ]).transpose(1, 0)
#     print(_infer_relative_position_token(input_ids, 1))
#     print(_infer_relative_position_turn(input_ids, 5))
"""

if __name__ == '__main__':
    # _relative_position_bucket_test()
    # _relative_position_bias_test()
    # _multi_head_attention_rpe_test()
    pass
