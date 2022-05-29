import math
import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.utils import move_to_cuda


@register_criterion('ved_loss')
class VEDLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        # ngram loss related
        self.eps = args.label_smoothing
        self.disable_ngram_loss = args.disable_ngram_loss
        # kl-loss related
        self.target_kl = args.target_kl
        self.kl_loss_weight = args.kl_loss_weight
        self.disable_kl_loss = args.disable_kl_loss
        # bag-of-word loss and masked-lm loss
        self.cls_bow_loss_weight = args.cls_bow_loss_weight
        self.latent_bow_loss_weight = args.latent_bow_loss_weight
        self.masked_lm_loss_weight = args.masked_lm_loss_weight
        # whether to use tfidf weights in bag-of-loss module
        self.use_tfidf_weights = args.use_tfidf_weights
        if self.use_tfidf_weights:
            assert args.tfidf_model_path is not None and args.tfidf_dictionary_path is not None
            self.tfidf_model_path = args.tfidf_model_path
            self.tfidf_dictionary_path = args.tfidf_dictionary_path
            self.tfidf_model = None
            self.tfidf_dictionary = None
            self.init_tfidf_model()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--disable-ngram-loss', action='store_true', default=False,
                            help='only compute basic stat')
        parser.add_argument('--use-tfidf-weights', action='store_true', default=False)
        parser.add_argument('--tfidf-model-path', type=str, help='tfidf model path')
        parser.add_argument('--tfidf-dictionary-path', type=str, help='tfidf dictionary path')

    def init_tfidf_model(self):
        from gensim.models import TfidfModel
        from gensim.corpora import Dictionary
        self.tfidf_model = TfidfModel.load(self.tfidf_model_path)
        self.tfidf_dictionary = Dictionary.load(self.tfidf_dictionary_path)
        print('| loading tfidf model from {} ...'.format(self.tfidf_model_path))

    def cal_tfidf_weights(self, targets, seq_len, eps=1e-4):
        # this function should input a torch tensor [batch, seq_len],
        # and output a tf-idf weight matrix torch tensor [batch, seq_len]
        # this function may effect the whole efficiency, since it can be pre-computed and cached ...
        _targets = targets.clone().cpu().numpy()
        tfidf_weights_map = [
            dict(tfidf_weight) for tfidf_weight in
            self.tfidf_model[[[(token_id, 1) for token_id in _target] for _target in _targets]]
        ]
        tfidf_weights = np.array(
            [[tfidf_weights_map[i].get(_item, eps) for _item in _target] for i, _target in enumerate(_targets)])
        # tfidf_weights = np.array([[weight[1] for weight in weights] for weights in self.tfidf_model[
        #     [[(token_id, 1) for token_id in _target] for _target in _targets]]])
        # use softmax will weaken the difference
        # tfidf_weights = F.softmax(tfidf_weights, dim=1)
        tfidf_weights = (tfidf_weights + eps) / (tfidf_weights.sum(axis=1)[:, None] + seq_len * eps) * seq_len
        tfidf_weights = move_to_cuda(torch.from_numpy(tfidf_weights.flatten('F')))
        return tfidf_weights

    def forward(self, model, sample, reduce=True):

        # execute forward method defined in model
        decoder_out, encoder_out = model(**sample['net_input'], return_all_hiddens=False)

        # n-gram predicting stream
        logits_list = decoder_out[0]

        # default targets fetch in sample directly
        targets = model.get_targets(sample, None)
        _, seq_len = targets.shape

        cls_bow_logits = encoder_out['cls_bow_logits']
        masked_logits = encoder_out['masked_logits']
        latent_bow_logits = encoder_out['latent_bow_logits']
        kl = encoder_out['kl']

        # # find tok k most possible words in response
        # value, index = F.log_softmax(bow_logits, dim=1)[0].topk(25, largest=False, dim=1)
        # print(self.task.target_dictionary.string(index))
        # # see the target method's probability
        # print(bow_logits[0][targets[0]])

        # calculate bag of word loss
        cls_bow_loss = None
        if cls_bow_logits is not None:
            cls_bow_lprobs = F.log_softmax(cls_bow_logits, dim=-1, dtype=torch.float32)
            cls_bow_loss = F.nll_loss(
                input=cls_bow_lprobs.repeat(seq_len, 1),
                target=targets.transpose(1, 0).contiguous().view(-1),
                reduction='sum', ignore_index=self.padding_idx)

        if self.use_tfidf_weights:
            assert latent_bow_logits is not None, 'if `use_tfidf_weights`, latent_bow_logits should not be None!'

        latent_bow_loss = None
        if latent_bow_logits is not None:
            _, seq_len = targets.shape
            latent_bow_lprobs = F.log_softmax(latent_bow_logits, dim=-1, dtype=torch.float32)
            if self.use_tfidf_weights:
                latent_bow_loss = F.nll_loss(
                    input=latent_bow_lprobs.repeat(seq_len, 1),
                    target=targets.transpose(1, 0).contiguous().view(-1),
                    reduction='none', ignore_index=self.padding_idx)
                latent_bow_loss = torch.sum(torch.mul(latent_bow_loss, self.cal_tfidf_weights(targets, seq_len)))
            else:
                latent_bow_loss = F.nll_loss(
                    input=latent_bow_lprobs.repeat(seq_len, 1),
                    target=targets.transpose(1, 0).contiguous().view(-1),
                    reduction='sum', ignore_index=self.padding_idx)

        masked_lm_loss = None
        if masked_logits is not None:
            masked_targets = sample['masked_target'].long()
            masked_lprobs = F.log_softmax(masked_logits, dim=-1, dtype=torch.float32)
            masked_lm_loss = F.nll_loss(
                input=masked_lprobs.view(-1, masked_lprobs.size(-1)),
                target=masked_targets.view(-1),
                reduction='sum', ignore_index=self.padding_idx)

        # calculate ngram predicting loss
        ngram = len(logits_list)
        expend_targets = targets.new_zeros(ngram, targets.size(0), targets.size(1)).fill_(self.padding_idx)
        for i in range(ngram):
            if i > 0 and self.disable_ngram_loss:
                break
            padding_targets = torch.zeros_like(targets).fill_(self.padding_idx)
            if 'target_idx' in sample:
                expend_targets[i, :, :] = torch.where(sample['target_idx'] >= i, targets, padding_targets)
            else:
                expend_targets[i, :, :] = targets
        targets = expend_targets

        # re-construction loss
        logits = torch.cat(logits_list, dim=0)
        lprobs = F.log_softmax(logits.view(-1, logits.size(-1)), dim=-1, dtype=torch.float32)
        rc_loss = F.nll_loss(
            input=lprobs,
            target=targets.view(-1),
            reduction='sum', ignore_index=self.padding_idx)

        if self.eps > 0.:
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
            non_pad_mask = targets.ne(self.padding_idx).view(-1)
            smooth_loss = smooth_loss[non_pad_mask]
            smooth_loss = smooth_loss.sum()
            eps_i = self.eps / lprobs.size(-1)
            # print(smooth_loss.item())
            rc_loss = (1. - self.eps) * rc_loss + eps_i * smooth_loss

        kl_loss, masked_kl_loss = None, None
        if not self.disable_kl_loss and kl is not None:
            kl_loss = kl.clone().sum()
            masked_kl = (kl > self.target_kl)
            masked_kl_loss = torch.mul(kl, masked_kl).sum()

        # total loss
        loss = rc_loss

        if cls_bow_loss is not None:
            loss = loss + self.cls_bow_loss_weight * cls_bow_loss
        if latent_bow_loss is not None:
            loss = loss + self.latent_bow_loss_weight * latent_bow_loss
        if masked_kl_loss is not None:
            loss = loss + self.kl_loss_weight * masked_kl_loss
        if masked_lm_loss is not None:
            loss = loss + self.masked_lm_loss_weight * masked_lm_loss

        sample_size = targets.ne(self.padding_idx).int().sum().item()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            # 'n_masked_tokens': sample.get('n_masked_tokens', None),
            'rc_loss': utils.item(rc_loss.data) if reduce else rc_loss.data,
        }

        if cls_bow_loss is not None:
            logging_output.update({
                'cls_bow_loss': utils.item(cls_bow_loss.data) if reduce else cls_bow_loss.data})
        if latent_bow_loss is not None:
            logging_output.update({
                'latent_bow_loss': utils.item(latent_bow_loss.data) if reduce else latent_bow_loss.data})
        if masked_kl_loss is not None:
            logging_output.update({
                'masked_kl_loss': utils.item(masked_kl_loss.data) if reduce else masked_kl_loss.data})
        if kl_loss is not None:
            logging_output.update({
                'kl_loss': utils.item(kl_loss.data) if reduce else kl_loss.data})
        if masked_lm_loss is not None:
            logging_output.update({
                'masked_lm_loss': utils.item(masked_lm_loss.data) if reduce else masked_lm_loss.data})
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):

        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        rc_loss = sum(log.get('rc_loss', 0) for log in logging_outputs)
        cls_bow_loss = sum(log.get('cls_bow_loss', 0) for log in logging_outputs)
        latent_bow_loss = sum(log.get('latent_bow_loss', 0) for log in logging_outputs)
        masked_kl_loss = sum(log.get('masked_kl_loss', 0) for log in logging_outputs)
        kl_loss = sum(log.get('kl_loss', 0) for log in logging_outputs)
        # n_masked_tokens = sum(log.get('n_masked_tokens', 0) for log in logging_outputs)
        # masked_lm_loss = sum(log.get('masked_lm_loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': rc_loss / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            # 'masked_lm_loss': 0 if n_masked_tokens == 0 else masked_lm_loss / n_masked_tokens,
            'rc_loss': rc_loss / sample_size / math.log(2),
            'masked_kl_loss': masked_kl_loss / nsentences,
            'kl_loss': kl_loss / nsentences,
            'latent_bow_loss': latent_bow_loss / sample_size / math.log(2),
            'cls_bow_loss': cls_bow_loss / sample_size / math.log(2),
        }

        return agg_output
