import os
import sys
sys.path.extend([
    '/home/v-wchen2/PycharmProjects/ProphetNet',
    '/mnt/c/Users/v-wchen2/PycharmProjects/ProphetNet',
    'C:/Users/v-wchen2/PycharmProjects/ProphetNet',
])

from utils.processor import PRETRAIN_PREFIX_PATH, split_to_shards

# PRETRAIN_PREFIX_PATH = '/mnt/dialogue/pretrain'

# to get this file <train.tsv>, follow <https://github.com/microsoft/DialoGPT>
REDDIT_IN_PATH = os.path.join(PRETRAIN_PREFIX_PATH, 'reddit/original_data/train.tsv')
REDDIT_OUT_PATH_FINETUNE = os.path.join(PRETRAIN_PREFIX_PATH, 'reddit/processed/finetune')
REDDIT_OUT_PATH_PRETRAIN = os.path.join(PRETRAIN_PREFIX_PATH, 'reddit/processed/pretrain')

# construct reddit sample data
# construct_reddit_sample(fin=REDDIT_IN_PATH, fout=os.path.join(PRETRAIN_PREFIX_PATH, 'reddit/processed/test'))

# TODO: calling this function will raises an error, see mp_process.py
# convert_reddit(REDDIT_IN_PATH, REDDIT_OUT_PATH_FINETUNE, REDDIT_OUT_PATH_PRETRAIN)

# TODO: check whether multiple shards are necessary
split_to_shards(
    fins=[os.path.join(REDDIT_OUT_PATH_FINETUNE, fin) for fin in ['train.src', 'train.tgt']],
    num_lines=215023265,
    num_shards=10
)
split_to_shards(
    fins=[os.path.join(REDDIT_OUT_PATH_FINETUNE, fin) for fin in ['valid.src', 'valid.tgt']],
    num_lines=1000000,
    num_shards=10
)


# find bugs
import numpy as np
from tqdm import tqdm


def infer_batch_size(path):
    with open(path, 'r') as f:
        lens = []
        for line in tqdm(f):
            lens.append(len(line.strip().split()))
    return np.array(lens)


train_src = infer_batch_size(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train.src'))
train_tgt = infer_batch_size(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train.tgt'))
valid_src = infer_batch_size(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid.src'))
valid_tgt = infer_batch_size(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid.tgt'))

# 99.9% of inputs have a length of 147, 99.9% of outputs have a length of 48
print(np.percentile(train_src, 99.9))
print(np.percentile(train_tgt, 99.9))
print(np.percentile(valid_src, 99.9))
print(np.percentile(valid_tgt, 99.9))


# filter
max_source_positions = 128
max_target_positions = 64
gap = 4


src = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train.src'), 'r')
tgt = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train.tgt'), 'r')
src_ = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train_.src'), 'w')
tgt_ = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'train_.tgt'), 'w')

num1, num2 = 0, 0

for src_lin, tgt_lin in tqdm(zip(src, tgt)):
    if len(src_lin.strip().split()) < max_source_positions - gap and \
            len(tgt_lin.strip().split()) < max_target_positions - gap:
        src_.write(src_lin)
        tgt_.write(tgt_lin)
        num1 += 1
    else:
        num2 += 1

print('write {} lines, discard {} lines'.format(num1, num2))


src = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid.src'), 'r')
tgt = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid.tgt'), 'r')
src_ = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid_.src'), 'w')
tgt_ = open(os.path.join(REDDIT_OUT_PATH_FINETUNE, 'valid_.tgt'), 'w')

num1, num2 = 0, 0

for src_lin, tgt_lin in tqdm(zip(src, tgt)):
    if len(src_lin.strip().split()) < max_source_positions - gap and \
            len(tgt_lin.strip().split()) < max_target_positions - gap:
        src_.write(src_lin)
        tgt_.write(tgt_lin)
        num1 += 1
    else:
        num2 += 1

print('write {} lines, discard {} lines'.format(num1, num2))


# split to train and valid
import random

REDDIT_PATH = ''
REDDIT_SPLIT_PATH = ''

src = open(os.path.join(REDDIT_PATH, 'train.src'), 'r')
tgt = open(os.path.join(REDDIT_PATH, 'train.tgt'), 'r')

train_src = open(os.path.join(REDDIT_PATH, 'train.src'), 'w')
train_tgt = open(os.path.join(REDDIT_PATH, 'train.tgt'), 'w')
valid_src = open(os.path.join(REDDIT_SPLIT_PATH, 'valid.src'), 'w')
valid_tgt = open(os.path.join(REDDIT_SPLIT_PATH, 'valid.tgt'), 'w')


for src_lin, tgt_lin in tqdm(zip(src, tgt)):
    if random.random() < .1:
        valid_src.write(src_lin)
        valid_tgt.write(tgt_lin)
    else:
        train_src.write(src_lin)
        train_tgt.write(tgt_lin)


########################################################################################################################
from tqdm import tqdm


REDDIT_PATH = '/home/v-wchen2/data/dialogue/pretrain/reddit/processed/finetune'
REDDIT_SPLIT_PATH = ''

src = open(os.path.join(REDDIT_PATH, 'train.src'), 'r')
tgt = open(os.path.join(REDDIT_PATH, 'train.tgt'), 'r')


src_large, tgt_large = [], []

for src_line, tgt_line in tqdm(zip(src, tgt)):
    if len(src_line.strip().split()) > 125 or len(src_line.strip().split()) > 63:
        src_large.append(src_line)
        tgt_large.append(tgt_line)


from sklearn.model_selection import train_test_split

src_large_train, src_large_valid, tgt_large_train, tgt_large_valid = train_test_split(
    src_large, tgt_large, test_size=0.05, random_state=0)


def write_lines(str_list, fout):
    with open(fout, 'w', encoding='utf-8') as f:
        for line in str_list:
            f.write(line)


write_lines(src_large_train, 'train.src')
write_lines(tgt_large_train, 'train.tgt')
write_lines(src_large_valid, 'valid.src')
write_lines(tgt_large_valid, 'valid.tgt')

