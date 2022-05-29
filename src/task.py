import os
import itertools

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
)

from .utils import BertDictionary
from .dataset import LanguagePairDatasetVED


def split_exists(split, src, tgt, lang, data_path, dataset_impl):
    filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
    return indexed_dataset.dataset_exists(filename, impl=dataset_impl)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, add_cls_to_source=False,
    mask_source=False, masked_prob=0.15, masked_span_len=2, min_masked_len=15,
    auto_infer_absolute_positions=False, auto_infer_relative_positions=False,
):

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        if split_exists(split_k, src, tgt, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path, dataset_impl):
            prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path, split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    return LanguagePairDatasetVED(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        append_bos=False,
        align_dataset=align_dataset,
        add_cls_to_source=add_cls_to_source,
        mask_source=mask_source,
        masked_prob=masked_prob,
        masked_span_len=masked_span_len,
        min_masked_len=min_masked_len,
        auto_infer_absolute_positions=auto_infer_absolute_positions,
        auto_infer_relative_positions=auto_infer_relative_positions,
    )


@register_task('ved_translate')
class DialogVEDTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)

        parser.add_argument('--add-cls-to-source', default=False, action='store_true',
                            help='whether to add [CLS] token to the begin of sentence or not, '
                                 'it\'s recommended to include in VAE-based models')
        parser.add_argument('--mask-source', default=False, action='store_true', help='whether to mask input or not')
        parser.add_argument('--masked-prob', type=float, default=0.15, help='masked probability')
        parser.add_argument('--masked-span-len', type=int, default=2, help='masked span length')
        parser.add_argument('--min-masked-len', type=int, default=15, help='minimal source length if masked')
        parser.add_argument('--auto-infer-absolute-positions', default=False, action='store_true',
                            help='whether to auto infer absolute positions')
        parser.add_argument('--auto-infer-relative-positions', default=False, action='store_true',
                            help='whether to auto infer relative positions')

    @classmethod
    def load_dictionary(cls, vocab_path: str):
        return BertDictionary.build_dictionary(vocab_path='vocab.txt', has_freq=False)

    @classmethod
    def setup_task(cls, args, **kwargs):

        paths = args.data.split(':')
        assert len(paths) > 0

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        d = cls.load_dictionary('vocab.txt')
        print('| dictionary: {} types'.format(len(d)))

        return cls(args, d, d)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            add_cls_to_source=self.args.add_cls_to_source,
            mask_source=self.args.mask_source,
            masked_prob=self.args.masked_prob,
            masked_span_len=self.args.masked_span_len,
            auto_infer_absolute_positions=self.args.auto_infer_absolute_positions,
            auto_infer_relative_positions=self.args.auto_infer_relative_positions,
        )

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions


@register_task('translation_prophetnet')
class DialogVEDTaskPure(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, vocab_path: str):
        return BertDictionary.build_dictionary('vocab.txt', False)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions
