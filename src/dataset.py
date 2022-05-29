import torch
from fairseq.data import LanguagePairDataset
from fairseq.data import data_utils
import numpy as np

from .utils import (
    _infer_absolute_position_sentence_backward,
    _infer_absolute_position_role_backward_with_knowledge,
    _infer_relative_position_token,
    _infer_relative_position_sentence,
)


def collate(
        samples, pad_idx, sep_idx, soc_idx, cls_idx, eos_idx,
        left_pad_source=True, left_pad_target=False,
        input_feeding=True, mask_source=False,
        auto_infer_absolute_positions=False,
        auto_infer_relative_positions=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        # we did a simple change on this merge function
        # to adapt to masked merge masked_indices and masked_tokens
        return data_utils.collate_tokens(
            [s[key] if s[key] is not None else torch.empty(0) for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(_alignments):
        align_tgt = _alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        _align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / _align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    # ------------------------------------------------------------------------------------------------------------------
    # add masked indices and tokens
    masked_tokens, masked_target = None, None

    if mask_source:
        # since the src_tokens is sorted by source length, if the first sample is not masked
        # it means that this batch is not masked because the max sentence length is still too short
        if samples[sort_order[0].item()].get('masked_tokens', None) is not None:
            masked_tokens = merge('masked_tokens', left_pad=left_pad_source)
            masked_target = merge('masked_target', left_pad=left_pad_source)
            masked_tokens = masked_tokens.index_select(0, sort_order)
            masked_target = masked_target.index_select(0, sort_order)

    # calculate total masked tokens
    n_masked_tokens = sum(len(
        s.get('masked_tokens')) if s.get('masked_tokens', None) is not None else 0 for s in samples)

    sentence_positions, role_positions = None, None

    # compute absolute position matrix
    if auto_infer_absolute_positions:
        sentence_positions = _infer_absolute_position_sentence_backward(
            src_tokens, sep_idx, pad_idx, cls_idx)
        role_positions = _infer_absolute_position_role_backward_with_knowledge(
            src_tokens, sep_idx, pad_idx, soc_idx, cls_idx)

    # compute relative position matrix
    relative_position_token, relative_position_sentence = None, None

    if auto_infer_relative_positions:
        relative_position_token = _infer_relative_position_token(src_tokens, pad_idx)
        relative_position_sentence = _infer_relative_position_sentence(src_tokens, sep_idx)

    # ------------------------------------------------------------------------------------------------------------------

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'n_masked_tokens': n_masked_tokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'sentence_positions': sentence_positions,
            'role_positions': role_positions,
            'relative_position_token': relative_position_token,
            'relative_position_sentence': relative_position_sentence,
            'masked_tokens': masked_tokens,
        },
        'target': target,
        'masked_target': masked_target,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


# dialogue corpus format during seq2seq/vae/rl pre-training:
# context: <s> <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>
# response: <s> <response> </s>

# *** Training ***
# encoder input: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
# <CLS> is used to predict bag of words in response
# encoder input: </s> <response>
# decoder output: <response> </s>
# </s> is added to the beginning due to fairseq convention, although I prefer `<s> <response>`

# *** Inference ***
# encoder input: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
# first decode input: </s>
# The decoder will use the predicted token as new input until output </s>

# if more special symbols need to be added, replace <unused_{}> with it in `vocab.txt` file
# compared to LanguagePairDataset, LanguagePairDatasetVAE add a parameter add_cls_to_source
# more customized operations are feasible


class LanguagePairDatasetVED(LanguagePairDataset):
    def __init__(
            self, src, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False,
            append_eos_to_target=False,
            align_dataset=None,
            append_bos=False,
            seed=1,
            # new added parameters
            add_cls_to_source=False, mask_source=False,
            masked_prob=0.15, masked_span_len=2, min_masked_len=15,
            auto_infer_absolute_positions=False,
            auto_infer_relative_positions=False,
    ):
        # add_cls_to_source: whether to add [CLS] token in the beginning of each example
        # masked_source: whether to mask source
        super().__init__(
            src, src_sizes, src_dict, tgt, tgt_sizes, tgt_dict, left_pad_source, left_pad_target,
            max_source_positions, max_target_positions, shuffle, input_feeding,
            remove_eos_from_source, append_eos_to_target, align_dataset, append_bos
        )
        self.add_cls_to_source = add_cls_to_source
        self.mask_source = mask_source
        self.masked_prob = masked_prob
        self.masked_span_len = masked_span_len
        self.auto_infer_absolute_position = auto_infer_absolute_positions
        self.auto_infer_relative_positions = auto_infer_relative_positions

        # if the input is too short, the mask is not executed
        self.min_masked_len = int(max(min_masked_len, int(1 / self.masked_prob * self.masked_span_len)))
        # the replace probs indicate the probability the masked token is replaced by:
        # 1. <MASK>  2. unchanged  3. random token in dictionary
        self.replace_probs = torch.tensor([0.8, 0.1, 0.1])

    def __getitem__(self, index):
        # with data_utils.numpy_seed(self.seed, self.epoch, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # clone them since there are some inplace operations below
        src_item, tgt_item = src_item.clone(), tgt_item.clone()

        if self.add_cls_to_source:
            # add [CLS] token at the beginning of each sentence
            src_item = torch.cat([torch.LongTensor([self.src_dict.cls()]), src_item])

        source_len = len(src_item)
        masked_tokens, masked_target = None, None

        if self.mask_source and source_len > self.min_masked_len:
            # output masked tokens and masked target
            # we do not mask [CLS], since:
            # 1). [CLS] is always the first token
            # 2). [CLS] is different from other tokens, since [CLS] need to understand the whole context
            # and connect to latent space
            masked_tokens = self.cal_mask_tokens(source_len)
            masked_target = src_item[masked_tokens].clone()
            # mask it
            src_item[masked_tokens] = self.replace(src_item[masked_tokens])

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'masked_tokens': masked_tokens,
            'masked_target': masked_target,
        }

        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def cal_mask_tokens(self, source_len):
        positions = np.arange(source_len)
        # each token has the same probability to be selected
        # masked_indices_start = positions[np.random.random(size=source_len) < self.masked_prob]
        # masked length, the longer the input, the longer the masked_len
        masked_len = min(int(round(self.masked_prob * source_len / self.masked_span_len)), 1)
        # there is no restriction on the special token
        # each token has the same probability to be masked
        # once a token is chosen, we extent this token to form a span, and mask the whole span
        masked_tokens = np.random.choice(positions, masked_len, replace=False)
        # extend this token to form a span to be the target to be predicted
        for step in range(1, self.masked_span_len):
            masked_tokens_end = masked_tokens + step
            masked_tokens = np.append(masked_tokens, masked_tokens_end)
        masked_tokens = np.sort(np.unique(masked_tokens[masked_tokens < source_len]))
        masked_tokens = torch.tensor(masked_tokens, dtype=torch.int64)
        return masked_tokens

    def replace(self, x):
        x_real = x.clone()
        x_rand = x.clone().random_(self.src_dict.nspecial, len(self.src_dict))
        x_mask = x.clone().fill_(self.src_dict.mask())
        # this sampling need a data_utils.torch_seed like function to control repeatability
        probs = torch.multinomial(self.replace_probs, len(x), replacement=True)
        masked = torch.LongTensor(x_mask * (probs == 0) + x_real * (probs == 1) + x_rand * (probs == 2))
        return masked

    def collater(self, samples):
        return collate(
            samples, pad_idx=self.src_dict.pad(), sep_idx=self.src_dict.sep(),
            soc_idx=self.src_dict.soc(), cls_idx=self.src_dict.cls(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, mask_source=self.mask_source,
            auto_infer_absolute_positions=self.auto_infer_absolute_position,
            auto_infer_relative_positions=self.auto_infer_relative_positions,
        )
