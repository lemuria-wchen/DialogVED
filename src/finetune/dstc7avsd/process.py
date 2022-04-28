import os
import sys
sys.path.extend(['/home/v-wchen2/PycharmProjects/ProphetNet', 'C:/Users/v-wchen2/PycharmProjects/ProphetNet'])

from src.utils.processor import FINETUNE_PREFIX_PATH, convert_dstc7_avsd, check


ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/original_data')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd/processed/drop_knowledge')


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
