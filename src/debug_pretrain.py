"""
import sys
sys.path.append('/home/v-wchen2/PycharmProjects/ProphetNet')

from src.prophetnet_dialog.bert_dictionary import BertDictionary
from src.prophetnet_dialog.masked_dataset import MaskedLanguagePairDataset

from fairseq.data import data_utils, TokenBlockDataset
import torch


split_path = '/home/v-wchen2/Data/dialogue/pretrain/reddit/binary_test/train'
dictionary = BertDictionary.load_from_file('src/vocab.txt')

dataset = data_utils.load_indexed_dataset(split_path, dictionary, 'mmap')

print(dataset[0])
print(dictionary.string(dataset[0]))
print(len(dataset[0]))
print(len(dictionary.string(dataset[0]).split(' ')))

dataset = TokenBlockDataset(dataset, dataset.sizes, 512, pad=dictionary.pad(), eos=dictionary.eos(), break_mode=None)

pred_probs = torch.FloatTensor([float(x) for x in [0.8, 0.1, 0.1]])

s2s_dataset = MaskedLanguagePairDataset(
    dataset, dataset.sizes, dictionary,
    shuffle=False, mask_prob=0.15,
    pred_probs=pred_probs,
)


from src.prophetnet_dialog.ngram_s2s_model import NgramTransformerProphetModel, transformer_big
from src.prophetnet_dialog.ngram_masked_s2s import NgramMaskedS2STask
from src.prophetnet_dialog.ngram_criterions import NgramLmLoss
from fairseq.trainer import Trainer
from fairseq import checkpoint_utils
import argparse
import os
import torch

import importlib

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


parser = argparse.ArgumentParser()
args = parser.parse_args()

base_dir = '/home/v-wchen2/Data/dialogue/'

args.data = os.path.join(base_dir, 'pretrain/reddit/binary_test')
args.load_from_pretrained_model = os.path.join(
    base_dir, 'pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt')

args.label_smoothing = 0.1
args.disable_ngram_loss = True
args.cpu = False
args.fp16 = False
args.fast_stat_sync = None
args.dataset_impl = 'mmap'
args.train_subset = 'train'
args.tokens_per_sample = 512
args.sample_break_mode = None
args.mask_s2s_mask_keep_rand = '0.8,0.1,0.1'
args.mask_s2s_prob = 0.15
args.seed = 1
args.distort_s2s_prob = 0.0
args.max_tokens = 2000
args.max_sentences = 8
args.distributed_world_size = 1
args.required_batch_size_multiple = False
args.distributed_rank = None
args.num_workers = 1

# # training
# SUFFIX='_pretrain_test'
# #BASE_DIR=/mnt/d/dialogue/pretrain/reddit
# BASE_DIR=/home/v-wchen2/Data/dialogue/pretrain/reddit
# SAVE_DIR=${BASE_DIR}/checkpoints${SUFFIX}
# TENSORBOARD_LOGDIR=${BASE_DIR}/tensorboard${SUFFIX}
# DATA_DIR=${BASE_DIR}/binary_test
#
# # parameters that do not require additional parameters
# USER_DIR=./prophetnet/
# ARCH=ngram_transformer_prophet_large
# CRITERION=ngram_language_loss
#
# fairseq-train ${DATA_DIR} \
#     --user-dir ${USER_DIR} --task ngram_masked_s2s --arch ${ARCH} \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
#     --lr 0.0003 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
#     --warmup-init-lr 1e-07 --warmup-updates 5000 \
#     --criterion $CRITERION \
#     --update-freq 8 --max-tokens 2500 --max-sentences 6 \
#     --num-workers 1  \
#     --dropout 0.1 --attention-dropout 0.1 \
#     --activation-dropout 0.0 --weight-decay 0.01 \
#     --save-dir ${SAVE_DIR} \
#     --max-epoch 10 \
#     --keep-last-epochs 10 \
#     --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
#     --dataset-impl mmap --empty-cache-freq 64 \
#     --seed 1 --mask-s2s-prob 0.15 \
#     --skip-invalid-size-inputs-valid-test \
#     --distributed-no-spawn

transformer_big(args)

task = NgramMaskedS2STask.setup_task(args)
criterion = NgramLmLoss.build_criterion(args, task)
model = NgramTransformerProphetModel.build_model(args, task)

trainer = Trainer(args, task, model, criterion, None)
extra_state, epoch_itr = checkpoint_utils.load_checkpoint(None, trainer)

a = trainer.get_train_iterator(epoch=0)
"""

# a complete FairSeq debug process for prophetnet

# import os
# from fairseq import utils
# from fairseq import options


# def add_default(_parser, _default_args):
#     for key in _default_args:
#         for action in _parser._actions:
#             if key == action.dest:
#                 action.required = False
#                 if _default_args.get(key):
#                     action.default = _default_args.get(key)
#
#
# def show_default(_parser, key):
#     for action in _parser._actions:
#         if key == action.dest:
#             print('{}: {}'.format(key, action.default))
#             break


# import argparse
# a = argparse.ArgumentParser()
# a.add_argument('--arch', '-a', default='fconv', metavar='ARCH', required=True)
# a.add_argument('data', default=20, type=int, metavar='NORM', help='clip threshold of gradients')
# print(a.format_help())


# for action in parser._actions:
#     if action.required:
#         print(action.option_strings, action.dest)
#
# parser.add_argument('dir', required=True)

# parser.add_argument('--arch', '-a', default='ngram_transformer_prophet_large')
# print(parser.format_help())
# add_default(parser, {
#     'arch': None,
# })

# show_default(parser, 'arch')

# show_default(parser)
# args.user_dir = './src/prophetnet_dialog'
# base_dir = '/home/v-wchen2/Data/dialogue/'
# args.data = os.path.join(base_dir, 'pretrain/reddit/binary_test')
# args.max_tokens = 2500
# args.max_sentences = 8
# args.task = 'ngram_masked_s2s'
# from fairseq_cli import train


# finally, I can debug fairseq-0.9.0 now.
# 1) add import sys; print(sys.argv) in fairseq_cli/train.py
# 2) edit get_training_parser, get_parser function in fairseq/options.py
# then you can get complete args, and follow the code bellow to get complete debug process

# a complete fairseq debug code for prophetnet
from fairseq import options, tasks, utils, checkpoint_utils
from fairseq.trainer import Trainer
from fairseq.meters import StopwatchMeter
import math
from fairseq.data import iterators

input_args = [
    '--data', '/home/v-wchen2/data/dialogue/pretrain/reddit/binary/finetune_sample',
    '--user-dir', './prophetnet_dialog/',
    '--task', 'ngram_masked_s2s',
    '--arch', 'ngram_transformer_prophet_large',
    '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)', '--adam-eps', '1e-6', '--clip-norm', '1.0',
    '--lr', '0.0003', '--lr-scheduler',
    'inverse_sqrt', '--min-lr', '1e-09', '--warmup-init-lr', '1e-07', '--warmup-updates', '5000',
    '--criterion', 'ngram_language_loss',
    '--update-freq', '8', '--max-tokens', '2500', '--max-sentences', '6', '--num-workers', '1',
    '--load-from-pretrained-model',
    '/home/v-wchen2/data/dialogue/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt',
    '--dropout', '0.1',
    '--attention-dropout', '0.1',
    '--activation-dropout', '0.0',
    '--weight-decay', '0.01',
    # '--sample-break-mode', 'complete_doc',
    '--skip-invalid-size-inputs-valid-test',
    '--save-dir', '/home/v-wchen2/data/dialogue/pretrain/reddit/checkpoints/test',
    '--max-epoch', '10', '--keep-last-epochs', '10',
    '--tensorboard-logdir', '/home/v-wchen2/data/dialogue/pretrain/reddit/tensorboard/test',
    '--dataset-impl', 'mmap', '--empty-cache-freq', '64', '--seed', '1',
    '--mask-s2s-prob', '0.15', '--skip-invalid-size-inputs-valid-test', '--distributed-no-spawn']

parser = options.get_training_parser(args=input_args)
args = options.parse_args_and_arch(parser, input_args=input_args)
utils.import_user_module(args)
print(args)

task = tasks.setup_task(args)
task.load_dataset('valid', combine=False, epoch=0)

# get data
print(task.dictionary.string(task.datasets['valid'][0]['source']))
print(task.datasets['valid'][0]['source'])

model = task.build_model(args)
criterion = task.build_criterion(args)

trainer = Trainer(args, task, model, criterion)


# extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
# max_epoch = args.max_epoch or math.inf
# max_update = args.max_update or math.inf
# lr = trainer.get_lr()
# train_meter = StopwatchMeter()
# train_meter.start()
# valid_subsets = args.valid_subset.split(',')
#
#
# update_freq = args.update_freq[epoch_itr.epoch - 1] \
#     if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
# itr = epoch_itr.next_epoch_itr(
#     fix_batches_to_gpus=args.fix_batches_to_gpus,
#     shuffle=(epoch_itr.epoch >= args.curriculum),
# )
# itr = iterators.GroupedIterator(itr, update_freq)

# for samples in itr:
#     print(samples)
#     log_output = trainer.train_step(samples)
#     # continue
#     # TODO
#     # loss, sample_size, logging_output = criterion(model, samples)
#     print(samples[0]['net_input']['src_tokens'].shape)
#     print(task.dictionary.string(samples[0]['net_input']['src_tokens'][0]))
#     print(samples[0]['net_input']['src_tokens'][0])
#     break
