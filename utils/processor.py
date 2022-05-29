from pytorch_transformers import BertTokenizer
import os
import numpy as np
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time


def infer_prefix(mode: str = 'finetune') -> str:
    assert mode in ['finetune', 'pretrain'], 'mode must be either `finetune` or `pretrain`'
    if mode == 'finetune':
        if sys.platform.startswith('win'):
            prefix = 'D:/dialogue/finetune'
        else:
            prefix = '/home/v-wchen2/data/dialogue/finetune'
    else:
        if sys.platform.startswith('win'):
            prefix = 'D:/dialogue/pretrain'
        else:
            prefix = '/home/v-wchen2/data/dialogue/pretrain'
    return prefix


def remove_or_makedir(f_path: str) -> None:
    if os.path.exists(f_path):
        os.remove(f_path)
    else:
        # make dir if not exit
        os.makedirs(os.path.dirname(f_path), exist_ok=True)


def make_dir(path):
    if isinstance(path, str):
        os.makedirs(path, exist_ok=True)
    elif isinstance(path, list):
        for p in path:
            make_dir(p)
    else:
        raise ValueError


def prepare(fin: str, src_fout: str = None, tgt_fout: str = None) -> tuple:
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    fin = open(fin, 'r', encoding='utf-8').readlines()
    if src_fout:
        remove_or_makedir(src_fout)
        src_fout = open(src_fout, 'w', encoding='utf-8')
    if tgt_fout:
        remove_or_makedir(tgt_fout)
        tgt_fout = open(tgt_fout, 'w', encoding='utf-8')
    return tok, fin, src_fout, tgt_fout


def split_line_base(line: str, tok: BertTokenizer) -> list:
    return [' '.join(tokens) for tokens in [
        tok.tokenize(sent.strip()) for sent in line.split('__eou__')]]


def split_line(
        line: str,
        tok: BertTokenizer,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = False,
        use_knowledge: bool = False) -> tuple:
    if not has_knowledge:
        assert not use_knowledge
    if has_knowledge and use_knowledge:
        knowledge, src, tgt = line.strip().split('\t')
        src_line = knowledge_sep.join(split_line_base(knowledge, tok)) + connect_sep + sep.join(
            split_line_base(src, tok))
    else:
        if has_knowledge and not use_knowledge:
            _, src, tgt = line.strip().split('\t')
        else:
            src, tgt = line.strip().split('\t')
        src_line = sep.join(split_line_base(src, tok))
    tgt_line = sep.join(split_line_base(tgt, tok))
    return src_line, tgt_line


# unit test for split_line function
def unit_test():
    fin = os.path.join(FINETUNE_PREFIX_PATH, 'dailydialog', 'original_data/dial.test')
    tok, fin, _, _ = prepare(fin)
    src_line, tgt_line = split_line(fin[3], tok)
    print('source line: {}\ntarget line: {}\n'.format(src_line, tgt_line))

    fin = os.path.join(FINETUNE_PREFIX_PATH, 'personachat', 'original_data/dial.test')
    tok, fin, _, _ = prepare(fin)
    src_line, tgt_line = split_line(fin[3], tok, has_knowledge=True, use_knowledge=True)
    print('source line: {}\ntarget line: {}\n'.format(src_line, tgt_line))

    fin = os.path.join(FINETUNE_PREFIX_PATH, 'dstc7avsd', 'original_data/dial.valid')
    tok, fin, _, _ = prepare(fin)
    src_line, tgt_line = split_line(fin[3], tok, sep=' ', knowledge_sep=' ', has_knowledge=True, use_knowledge=True)
    print('source line: {}\ntarget line: {}\n'.format(src_line, tgt_line))


def convert_daily_dialog(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = False,
        use_knowledge: bool = False,
        max_src_pos: int = 512,
        max_tgt_pos: int = 128,
        prune: bool = True) -> None:
    tok, fin, src_fout, tgt_fout = prepare(fin, src_fout, tgt_fout)
    num_prunes, num_src_prunes, num_tgt_prunes = 0, 0, 0
    for line in tqdm(fin):
        src_line, tgt_line = split_line(
            line, tok, sep, knowledge_sep, connect_sep, has_knowledge, use_knowledge)
        src_line, tgt_line = src_line.split(), tgt_line.split()
        if len(src_line) > max_src_pos or len(tgt_line) > max_tgt_pos:
            num_prunes += 1
            num_src_prunes += len(src_line) > max_src_pos
            num_tgt_prunes += len(tgt_line) > max_tgt_pos
        src_line = ' '.join(src_line[: max_src_pos - 1]) if prune else ' '.join(src_line)
        tgt_line = ' '.join(tgt_line[: max_tgt_pos - 1]) if prune else ' '.join(tgt_line)
        src_fout.write('{}\n'.format(src_line))
        tgt_fout.write('{}\n'.format(tgt_line))
    src_fout.close()
    tgt_fout.close()
    print('{} lines exceed max positions, {} source lines and {} target lines have been pruned'.format(
        num_prunes, num_src_prunes, num_tgt_prunes))


def convert_persona_chat(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = True,
        use_knowledge: bool = True,
        max_src_pos: int = 512,
        max_tgt_pos: int = 128,
        prune: bool = True):
    convert_daily_dialog(fin, src_fout, tgt_fout, sep, knowledge_sep, connect_sep, has_knowledge, use_knowledge,
                         max_src_pos, max_tgt_pos, prune)


def convert_dstc7_avsd(
        fin: str,
        src_fout: str,
        tgt_fout: str,
        sep: str = ' [SEP] ',
        knowledge_sep: str = ' [SEP] ',
        connect_sep: str = ' [CLS] ',
        has_knowledge: bool = True,
        use_knowledge: bool = True,
        max_src_pos: int = 512,
        max_tgt_pos: int = 128,
        prune: bool = True,
        multi_refs_tgt_fout: str = None) -> None:
    if multi_refs_tgt_fout is not None:
        tok, fin, src_fout, tgt_fout = prepare(fin, src_fout, tgt_fout)
        remove_or_makedir(multi_refs_tgt_fout)
        multi_refs_tgt_fout = open(multi_refs_tgt_fout, 'w', encoding='utf-8')
        num_prunes, num_src_prunes, num_tgt_prunes = 0, 0, 0
        for line in tqdm(fin):
            src_line, tgt_lines = split_line(
                line, tok, sep, knowledge_sep, connect_sep, has_knowledge, use_knowledge)
            tgt_lines = [ref.strip() for ref in tgt_lines.split('|')]
            src_line, tgt_line = src_line.split(), tgt_lines[0].split()
            if len(src_line) > max_src_pos or len(tgt_line) > max_tgt_pos:
                num_prunes += 1
                num_src_prunes += len(src_line) > max_src_pos
                num_tgt_prunes += len(tgt_line) > max_tgt_pos
            src_line = ' '.join(src_line[: max_src_pos - 1]) if prune else ' '.join(src_line)
            tgt_line = ' '.join(tgt_line[: max_tgt_pos - 1]) if prune else ' '.join(tgt_line)
            src_fout.write('{}\n'.format(src_line))
            tgt_fout.write('{}\n'.format(tgt_line))
            multi_refs_tgt_fout.write('{}\n\n'.format('\n'.join(tgt_lines)))
        multi_refs_tgt_fout.close()
        print('{} lines exceed max positions, {} source lines and {} target lines have been pruned'.format(
            num_prunes, num_src_prunes, num_tgt_prunes))
    else:
        convert_daily_dialog(fin, src_fout, tgt_fout, sep, knowledge_sep, connect_sep, has_knowledge, use_knowledge,
                             max_src_pos, max_tgt_pos, prune)


def check(processed_path: str, mode: str = 'test') -> None:
    with open(os.path.join(processed_path, '{}.src'.format(mode)), encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(os.path.join(processed_path, '{}.tgt'.format(mode)), encoding='utf-8') as f:
        tgt_lines = f.readlines()
    max_src, max_tgt, max_src_tgt, max_tgt_src = -1, -1, -1, -1
    for src, tgt in zip(src_lines, tgt_lines):
        src_len = len(src.strip().split())
        tgt_len = len(tgt.strip().split())
        if src_len > max_src:
            max_src = src_len
            max_src_tgt = tgt_len
        if tgt_len > max_tgt:
            max_tgt = tgt_len
            max_tgt_src = src_len
    print('{}\nboundary shape src: ({}, {})\nboundary shape tgt: ({}, {})\n'.format(
        processed_path, max_src, max_src_tgt, max_tgt_src, max_tgt))


def tokenize_reddit(line: str, tok: BertTokenizer) -> list:
    _, contexts, response = line.strip().split('\t')
    contexts = [' '.join(
        tok.tokenize(' '.join(context.strip().split()[1:]))) for context in contexts.split('EOS')]
    response = ' '.join(tok.tokenize(' '.join(response.split(' ')[1:])))
    return contexts + [response]


def convert_for_finetune(arrays: list, fout: str, mode: str = 'train') -> None:
    fout_src = open(os.path.join(fout, '{}.src'.format(mode)), 'w', encoding='utf-8')
    fout_tgt = open(os.path.join(fout, '{}.tgt'.format(mode)), 'w', encoding='utf-8')
    for array in arrays:
        fout_src.write(' [SEP] '.join(array[: -1]) + '\n')
        fout_tgt.write(array[-1] + '\n')


def convert_for_pretrain(arrays: list, fout: str, mode: str = 'train') -> None:
    fout = open(os.path.join(fout, '{}.src'.format(mode)), 'w', encoding='utf-8')
    for array in arrays:
        fout.write(' [SEP] '.join(array) + '\n')


def convert_reddit(
        fin: str,
        finetune_path: str,
        pretrain_path: str,
        memory_enough: bool = False,
        test_size: int = 0.1,
        seed: int = 123) -> None:

    begin = time.time()
    fin = open(fin, 'r', encoding='utf-8').readlines()
    end = time.time()
    num_lines = len(fin)
    print('success to load original reddit data to memory, {} minutes used, {} lines in total'.format(
        round((end - begin) / 60, 4), num_lines))
    tok = BertTokenizer.from_pretrained('bert-base-uncased')

    # this operation requires too much memory and another way should be tried in small memory machine
    os.makedirs(finetune_path, exist_ok=True)
    os.makedirs(pretrain_path, exist_ok=True)

    if memory_enough:
        dialogs = []
        for line in tqdm(fin):
            dialogs.append(tokenize_reddit(line, tok))
        train_dialogs, valid_dialogs = train_test_split(dialogs, test_size=0.1, random_state=seed)
        # fine-tune / pretrain
        convert_for_finetune(train_dialogs, fout=finetune_path, mode='train')
        convert_for_finetune(valid_dialogs, fout=finetune_path, mode='valid')
        convert_for_pretrain(train_dialogs, fout=pretrain_path, mode='train')
        convert_for_pretrain(valid_dialogs, fout=pretrain_path, mode='valid')
        end = time.time()
        print('success to write processed reddit data to disk, {} minutes used'.format(round((end - begin) / 60, 4)))
    else:
        pretrain_train_src = open(os.path.join(pretrain_path, 'train.src'), 'w', encoding='utf-8')
        pretrain_valid_src = open(os.path.join(pretrain_path, 'valid.src'), 'w', encoding='utf-8')
        finetune_train_src = open(os.path.join(finetune_path, 'train.src'), 'w', encoding='utf-8')
        finetune_train_tgt = open(os.path.join(finetune_path, 'train.tgt'), 'w', encoding='utf-8')
        finetune_valid_src = open(os.path.join(finetune_path, 'valid.src'), 'w', encoding='utf-8')
        finetune_valid_tgt = open(os.path.join(finetune_path, 'valid.tgt'), 'w', encoding='utf-8')
        np.random.seed(seed)
        for line in tqdm(fin):
            arrays = tokenize_reddit(line, tok)
            if np.random.rand() < test_size:
                # valid set for pretrain
                pretrain_valid_src.write(' [SEP] '.join(arrays) + '\n')
                # train set for finetune
                for i in range(1, len(arrays)):
                    src, tgt = arrays[: i], arrays[i]
                    finetune_valid_src.write(' [SEP] '.join(src) + '\n')
                    finetune_valid_tgt.write(tgt + '\n')
            else:
                # train set for pretrain
                pretrain_train_src.write(' [SEP] '.join(arrays) + '\n')
                # train set for finetune
                for i in range(1, len(arrays)):
                    src, tgt = arrays[: i], arrays[i]
                    finetune_train_src.write(' [SEP] '.join(src) + '\n')
                    finetune_train_tgt.write(tgt + '\n')
        end = time.time()
        print('success to write processed reddit data to disk, {} minutes used'.format(round((end - begin) / 60, 4)))


def split_to_shards(fins: list, num_lines: int, num_shards: int = 5):
    import random
    if isinstance(fins, str):
        fins = [fins]

    fouts = [[fin.split('.')[0] + '.part{}.'.format(part) + fin.split('.')[1] for
              part in range(1, num_shards + 1)] for fin in fins]

    for fout in sum(fouts, []):
        remove_or_makedir(fout)
    fouts = [[open(fout, 'w', encoding='utf-8') for fout in _] for _ in fouts]

    fins = [open(fin, 'r', encoding='utf-8') for fin in fins]

    for _ in tqdm(range(num_lines)):
        lines = [fin.readline() for fin in fins]
        i = random.randint(0, len(fouts[0]) - 1)
        [fouts[j][i].write(line) for j, line in enumerate(lines)]

    for _ in fins:
        _.close()

    for _ in fouts:
        for __ in _:
            __.close()


def write_arrays(text_array: list, output_path: str) -> None:
    remove_or_makedir(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(text_array):
            f.write(text + '\n')


def construct_reddit_sample(fin: str, fout: str):
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    fin = open(fin, 'r', encoding='utf-8')

    single_turn_count, single_turn_max, single_turn_text = 0, 7000, []
    multi_turn_count, multi_turn_max, multi_turn_text = 0, 5000, []

    # generate sample reddit data
    for line in fin:
        # tokenize
        dialog = tokenize_reddit(line, tok)
        # control text
        if len(dialog) == 2 and single_turn_count < single_turn_max:
            single_turn_text.append(' [SEP] '.join(dialog))
            single_turn_count += 1
        if len(dialog) > 2 and multi_turn_count < multi_turn_max:
            multi_turn_text.append(' [SEP] '.join(dialog))
            multi_turn_count += 1
        if single_turn_count == single_turn_max and multi_turn_count == multi_turn_max:
            break

    text = single_turn_text + multi_turn_text
    train_text, valid_text = train_test_split(text, test_size=0.2, random_state=123)

    write_arrays(train_text, os.path.join(fout, 'train.src'))
    write_arrays(valid_text, os.path.join(fout, 'valid.src'))


FINETUNE_PREFIX_PATH = infer_prefix(mode='finetune')
PRETRAIN_PREFIX_PATH = infer_prefix(mode='pretrain')


if __name__ == '__main__':
    unit_test()
