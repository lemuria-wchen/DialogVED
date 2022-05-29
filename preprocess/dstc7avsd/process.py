import os
import sys
sys.path.append('/remote-home/wchen/project/DialogVED')
# sys.path.append('/Users/lemuria_chen/PycharmProjects/DialogVED')

from utils.processor import convert_dstc7_avsd, check


# FINETUNE_PREFIX_PATH = '/remote-home/wchen/project/DialogVED/data/finetune'
FINETUNE_PREFIX_PATH = '/Users/lemuria_chen/PycharmProjects/DialogVED/data/finetune'


ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed')


convert_dstc7_avsd(
    fin=os.path.join(ORIGINAL_PATH, 'dial.train'),
    src_fout=os.path.join(PROCESSED_PATH, 'train.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'train.tgt'),
    use_knowledge=False
)
convert_dstc7_avsd(
    fin=os.path.join(ORIGINAL_PATH, 'dial.valid'),
    src_fout=os.path.join(PROCESSED_PATH, 'valid.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'valid.tgt'),
    use_knowledge=False
)
convert_dstc7_avsd(
    fin=os.path.join(ORIGINAL_PATH, 'dial.test'),
    src_fout=os.path.join(PROCESSED_PATH, 'test.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'test.tgt'),
    multi_refs_tgt_fout=os.path.join(PROCESSED_PATH, 'test_multi_refs.tgt'),
    use_knowledge=False
)

check(PROCESSED_PATH, mode='train')
check(PROCESSED_PATH, mode='valid')
check(PROCESSED_PATH, mode='test')
