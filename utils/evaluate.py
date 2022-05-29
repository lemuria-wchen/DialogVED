from collections import Counter

from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import argparse


# references:
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/plato/metrics/metrics.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/dstc7_avsd_eval.py
# https://github.com/lemuria-wchen/Research/blob/master/NLP/Dialogue-PLATO/tools/knowledge_f1.py

# This script integrates all evaluation methods proposed in the Plato article.
# ACL 2020: https://www.aclweb.org/anthology/2020.acl-main.9.pdf
# Repository: https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO


def detokenize_bert(string: str) -> str:
    return string.replace(' ##', '')


# hyps and refs data loader for dailydialog & personachat
def load_file(hyps_file_path: str, refs_file_path=None, ignore_indices=None) -> tuple:
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
        hyps = [line.strip().split() for line in hyps_fin]
    refs = None
    if refs_file_path:
        with open(refs_file_path, 'r', encoding='utf-8') as f:
            refs_fin = f.readlines()
            if ignore_indices:
                assert isinstance(ignore_indices, list)
                refs_fin = [line for idx, line in enumerate(refs_fin) if idx not in ignore_indices]
        refs = [detokenize_bert(line.strip()).split() for line in refs_fin]
    return hyps, refs


# hyps and refs data loader for dstc7avsd
def _load_file(hyps_file_path: str, refs_file_path=None) -> tuple:
    # load predicted file and reference file
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
        hyps = [line.strip() for line in hyps_fin]
    with open(refs_file_path, 'r', encoding='utf-8') as f:
        refs, tmp = [], []
        for line in f.readlines():
            if line != '\n':
                tmp.append(detokenize_bert(line.strip()))
            else:
                refs.append(tmp)
                tmp = []
    assert len(hyps) == len(refs), 'number of instances of hyps and refs muse be equal'
    return hyps, refs


# calculate BLEU-1/2 for dailydialog & personachat
def bleu(hyps_file_path: str, refs_file_path: str, ignore_indices=None, output_path=None) -> tuple:
    hyps, refs = load_file(hyps_file_path, refs_file_path, ignore_indices)
    bleu_1, bleu_2 = [], []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[1, 0, 0, 0])
        except Exception as e:
            print(e)
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp, smoothing_function=SmoothingFunction().method7, weights=[0.5, 0.5, 0, 0])
        except Exception as e:
            print(e)
            score = 0
        bleu_2.append(score)
    bleu_1, bleu_2 = np.average(bleu_1), np.average(bleu_2)
    output_content = 'BLEU-1/2: {}/{}\n'.format(round(bleu_1, 4), round(bleu_2, 4))
    print('-------------- BLEU score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return bleu_1, bleu_2


# calculate Distinct-1/2 for dailydialog & personachat
def distinct(hyps_file_path: str, output_path=None) -> tuple:
    hyps, _ = load_file(hyps_file_path)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for hyp in hyps:
        unigrams = Counter(hyp)
        bigrams = Counter(zip(hyp, hyp[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(hyp)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(hyp)-1)+1e-5))
        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)
    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    output_content = 'Distinct-1/2: {}/{}\n'.format(round(inter_dist1, 4), round(inter_dist2, 4))
    print('-------------- Distinct score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


# calculate knowledge f1 for personachat
def knowledge_f1(hyps_file_path: str, refs_file_path: str, output_path=None) -> tuple:
    # load stopwords
    stopwords = set()
    with open('./stopwords.txt', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            stopwords.add(word)
    # load predicted file and reference file
    with open(hyps_file_path, 'r', encoding='utf-8') as f:
        hyps_fin = f.readlines()
    with open(refs_file_path, 'r', encoding='utf-8') as f:
        refs_fin = f.readlines()
    hyps = [line.strip() for line in hyps_fin]
    refs = [line.strip() for line in refs_fin]
    assert len(hyps) == len(refs), 'number of instances of hyps and refs muse be equal'
    # calculate knowledge f1 value
    cnt, res, r, p = 0, .0, .0, .0
    for hyp, ref in zip(hyps, refs):
        cnt += 1
        # prediction
        hyp = set(hyp.split())
        hyp = hyp - stopwords
        hyp_len = len(hyp)
        # reference
        knowledge, _, _ = ref.strip().split('\t')
        words = set()
        for sent in knowledge.split(" __eou__ "):
            for word in sent.split():
                words.add(word)
        words = words - stopwords
        k_len = len(words)
        overlap = len(words & hyp)
        if overlap == 0:
            continue
        recall = float(overlap) / k_len
        r += recall
        precision = float(overlap) / hyp_len
        p += precision
        res += 2 * recall * precision / (recall + precision)
    # recall/precision/f1
    output_content = 'Knowledge R/P/F1: {}/{}/{}\n'.format(round(r / cnt, 4), round(p / cnt, 4), round(res / cnt, 4))
    print('-------------- Knowledge score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return r / cnt, p / cnt, res / cnt


# calculate BLEU-1/2/3/4, METEOR, ROUGH-L and CIDEr for dstc7avsd
def score_fn(_hyp: dict, _ref: dict):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    _scores = {}
    for scorer, method in scorers:
        _score, _ = scorer.compute_score(_ref, _hyp)
        if type(_score) == list:
            for m, s in zip(method, _score):
                _scores[m] = s
        else:
            _scores[method] = _score
    return _scores


# calculate BLEU-1/2/3/4, METEOR, ROUGH-L and CIDEr for dstc7avsd
def coco_eval(hyps_file_path: str, refs_file_path: str, output_path=None) -> dict:
    hyps, refs = _load_file(hyps_file_path, refs_file_path)
    hyps = {idx: [hyp] for idx, hyp in enumerate(hyps)}
    refs = {idx: ref for idx, ref in enumerate(refs)}
    res = score_fn(hyps, refs)
    output_content = ''
    for name in res:
        output_content += '{}: {}\n'.format(name, round(res[name], 4))
    print('-------------- Microsoft COCO score --------------\n{}'.format(output_content))
    if output_path is not None:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(output_content)
    return res


def evaluate(task_name: str, hyps_file_path: str, refs_file_path: str, output_path: str, knowledge_file_path=None):

    assert task_name in ['dailydialog', 'personachat', 'dstc7avsd'], \
        'now only the evaluation of the above tasks is supported.'

    if task_name == 'dailydialog':
        bleu(hyps_file_path, refs_file_path, output_path=output_path)
        distinct(hyps_file_path, output_path=output_path)
    elif task_name == 'personachat':
        bleu(hyps_file_path, refs_file_path, output_path=output_path)
        distinct(hyps_file_path, output_path=output_path)
        assert knowledge_file_path is not None, 'if evaluate personachat, knowledge file path must be provided'
        knowledge_f1(hyps_file_path, knowledge_file_path, output_path=output_path)
    else:
        coco_eval(hyps_file_path, refs_file_path, output_path=output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='command line parameter for dialogue generation evaluation')

    parser.add_argument('-name', '--task_name', type=str, required=True, help='specify which task to evaluate')
    parser.add_argument('-hyp', '--hyps_file_path', type=str, required=True, help='predicted file path')
    parser.add_argument('-ref', '--refs_file_path', type=str, required=True, help='gold file path')
    parser.add_argument('-out', '--output_path', type=str, required=False, default=None, help='output path')
    parser.add_argument('-know', '--knowledge_file_path', type=str, required=False, default=None, help='knowledge path')
    args = parser.parse_args()

    evaluate(
        task_name=args.task_name, hyps_file_path=args.hyps_file_path,
        refs_file_path=args.refs_file_path, output_path=args.output_path,
        knowledge_file_path=args.knowledge_file_path,
    )
