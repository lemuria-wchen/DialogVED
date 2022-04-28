import torch

from fairseq.data import Dictionary
from fairseq.utils import make_positions


# special token id
# cls_id -> 4 | sep_id -> 5 | pad_id -> 1 | soc_id -> 9
PAD_INDEX = 0
KNOWLEDGE_ROLE = 3
MAX_SENTENCE = 31


"""
dialogue presentation during masked span pre-training:
<s> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>

1. in a masked turn, a complete turn will be masked as [MASK1]
2. in unmasked turn, some tokens can be masked as [MASK2]

dialogue presentation during seq2seq pre-training:
context: <s> <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> </s>
response: <s> <turn> </s>
if more special symbols need to be added, add it in vocab.txt directly
"""


# this dictionary module runs through the whole process of pre-processing, training and inference
class BertDictionary(Dictionary):
    def __init__(
            self,
            pad='[PAD]',
            eos='</s>',
            unk='[UNK]',
            bos='<s>'
    ):
        super().__init__(pad=pad, eos=eos, unk=unk, bos=bos)

        self.cls_word = '[CLS]'
        self.sep_word = '[SEP]'
        self.mask_word = '[MASK]'
        self.mask1_word = '[MASK1]'
        self.mask2_word = '[MASK2]'
        # start of conversation, text span before this id is seen as knowledge
        self.soc_word = '[SOC]'

    @classmethod
    def build_dictionary(cls, vocab_path: str, has_freq: bool):
        # bert dictionary from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
        d = cls()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            if has_freq:
                # each line is one token and it's frequency separated by space
                for line in f.readlines():
                    word, freq = line.strip().split()
                    d.add_symbol(word=word, n=freq)
            else:
                # each line is one token
                for line in f.readlines():
                    word = line.strip()
                    d.add_symbol(word=word, n=1)
        d.nspecial = 999
        return d

    def cls(self):
        return self.index(self.cls_word)

    def sep(self):
        return self.index(self.sep_word)

    def pad(self):
        assert self.index(self.pad_word) == self.pad_index
        return self.pad_index

    def mask(self):
        return self.index(self.mask_word)

    def mask1(self):
        return self.index(self.mask1_word)

    def mask2(self):
        return self.index(self.mask2_word)

    def soc(self):
        return self.index(self.mask2_word)


def _infer_absolute_position_sentence_forward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0      2      0      3      0      4        5
    # all token not belong to one specific turn can be seen as pad

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = torch.cumsum(input_ids.eq(sep_id).int(), dim=1) + 1
    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    # prevent array subscript out of index
    positions[positions > MAX_SENTENCE] = MAX_SENTENCE

    return positions


def _infer_absolute_position_sentence_backward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3>  <response>
    # index:       0      4      0      3      0      2         1
    # all token not belong to one specific turn can be seen as pad

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 2
    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    # prevent array subscript out of index
    positions[positions > MAX_SENTENCE] = MAX_SENTENCE

    return positions


def _infer_absolute_position_role_forward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0       2     0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = (torch.cumsum(input_ids.flip(1).eq(sep_id).int(), dim=1).flip(1) + 1) % 2 + 1

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX

    return positions


def _infer_absolute_position_role_backward(input_ids, sep_id, pad_id, cls_id):

    # given context: <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #
    #            <CLS> <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # index:       0      1      0       2     0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = torch.cumsum(input_ids.eq(sep_id).int(), dim=1) % 2

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX

    return positions


# this function is compatible
def _infer_absolute_position_role_backward_with_knowledge(input_ids, sep_id, pad_id, soc_id, cls_id):

    # we need a [SOT] token here to indicate the start of conversation
    # given context: <CLS> <know1> [SEP] <know2> [SOT] <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4>
    # response: <response> </s>
    #            <CLS> <know1> [SEP] <know2> [SOT] <turn1> [SEP] <turn2> <SEP> <turn3> <SEP> <turn4> <response>
    # output:      0      3      0      3      3      1      0      2      0      1      0      2        1

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ])

    positions = (torch.cumsum(input_ids.flip(1).eq(sep_id), dim=1).flip(1) + 1) % 2 + 1

    # add KNOWLEDGE_ROLE, the knowledge span should always be the left of soc_id
    # [SOC] means the start of conversation
    alpha = torch.cumsum(input_ids.flip(1).eq(soc_id), dim=1).flip(1)
    positions = (1 - alpha) * positions + alpha * KNOWLEDGE_ROLE

    positions[input_ids == cls_id] = PAD_INDEX
    positions[input_ids == pad_id] = PAD_INDEX
    positions[input_ids == sep_id] = PAD_INDEX
    positions[input_ids == soc_id] = PAD_INDEX

    return positions


def _infer_relative_position_token(input_ids, pad_id):
    """
    :param input_ids: (seq_len, batch_size)
    :param pad_id: <pad> index in the dictionary
    :return: token level relative position matrix before bucket
    """

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ]).transpose(1, 0)

    positions = make_positions(input_ids, padding_idx=pad_id).transpose(1, 0)

    # alpha = positions.eq(pad_id)
    # positions = (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)
    # alpha = (alpha.unsqueeze(0) + alpha.unsqueeze(1) + 0).permute(2, 0, 1)
    # positions = (1 - alpha) * positions

    return (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)


def _infer_relative_position_sentence(input_ids, sep_id):
    """
    a three-turns dialogue input sequence ids is supposed to be:
        <cls> <turn1> <sep> <turn2> <sep> <turn3>
    :param input_ids: (seq_len, batch_size)
    :param sep_id: <sep> index in the dictionary
    :return: turn level relative position matrix before bucket
    """

    # input_ids = torch.tensor([
    #     [4, 101, 191, 218, 5, 224, 241, 260, 179, 9, 361, 730, 259, 5, 491, 429, 407, 395, 1],
    #     [4, 101, 191, 229, 228, 9, 261, 270, 819, 5, 631, 830, 929, 5, 415, 402, 490, 1, 1],
    # ]).transpose(1, 0)

    positions = torch.cumsum(input_ids.transpose(1, 0).eq(sep_id).int(), dim=0)

    return (positions.unsqueeze(0) - positions.unsqueeze(1)).permute(2, 0, 1)
