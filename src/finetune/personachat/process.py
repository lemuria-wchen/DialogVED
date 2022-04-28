import os
import sys
sys.path.extend(['/home/v-wchen2/PycharmProjects/ProphetNet', 'C:/Users/v-wchen2/PycharmProjects/ProphetNet'])

from src.utils.processor import FINETUNE_PREFIX_PATH, convert_persona_chat, check


ORIGINAL_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'personachat/original_data')
PROCESSED_PATH = os.path.join(FINETUNE_PREFIX_PATH, 'personachat/processed/drop_knowledge')


convert_persona_chat(
    fin=os.path.join(ORIGINAL_PATH, 'dial.train'),
    src_fout=os.path.join(PROCESSED_PATH, 'train.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'train.tgt'),
    use_knowledge=False
)
convert_persona_chat(
    fin=os.path.join(ORIGINAL_PATH, 'dial.valid'),
    src_fout=os.path.join(PROCESSED_PATH, 'valid.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'valid.tgt'),
    use_knowledge=False
)
convert_persona_chat(
    fin=os.path.join(ORIGINAL_PATH, 'dial.test'),
    src_fout=os.path.join(PROCESSED_PATH, 'test.src'),
    tgt_fout=os.path.join(PROCESSED_PATH, 'test.tgt'),
    use_knowledge=False
)

check(PROCESSED_PATH, mode='train')
check(PROCESSED_PATH, mode='valid')
check(PROCESSED_PATH, mode='test')
