# DialogVED

Code and released pre-trained model for our ACL 2022 paper: [DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation](https://aclanthology.org/2022.acl-long.333/).

 
### News

- Fixed bugs in dailydialog, updated new training and evaluation scripts. (2022.06.19)
- Optimize code structure and remove redundant code. (2022.05.29) 
- Pretrained checkpoints of DialogVED have been released! (2022.05.17)

### TODO

- A fp16 version of DialogVED will be released, about 700M in size.
- Pre-trained scripts are scheduled to be released.

### Requirements

- python==3.7
- torch==1.3.0
- fairseq==0.9.0
- tensorboardX==1.7
- pytorch_transformers
- sklearn
- nltk==3.5

```shell
sudo apt install default-jdk
curl https://install.meteor.com/ | sh

pip install -r requirements.txt
```

### Pre-trained Models

We have released the following checkpoints for pre-trained models as described in the paper of [DialogVED](https://aclanthology.org/2022.acl-long.333/). Download the pre-trained checkpoint and set the `load-from-pretrained-model` parameter in the fine-tuning running command.  

- [DialogVED-VAE-Standard](https://drive.google.com/file/d/1EucujAl8vXyrEDAyAb0SeLovzIX2_9tn/view?usp=sharing)
- [DialogVED-VAE-Large](https://drive.google.com/file/d/1GLMrNAc2YEPJ-eiRcbFHGP0XGzAwKikM/view?usp=sharing)
- [DialogVED-Seq2Seq](https://drive.google.com/file/d/1xiRMBPeaIUvKFbnKrf7etXPVyU1C1x56/view?usp=sharing)

**Note**: DialogVED-VAE-Standard has a size of latent size 32, where DialogVED-VAE-Large has a size of latent size 64. DialogVED-Seq2Seq has no latent variable, it's a pure seq2seq model with the same training setting like DialogVED. It may perform better in scenarios where diversity of responses is less important.   

### Fine-tuning on your own dialogue datasets!

#### Data preparation

Prepare your train.src, train.tgt, dev.src, dev.tgt, test.src, test.tgt as follows, context and response of one dialogue sample are placed in the .src and .tgt file with one line. Use '[SEP] to separate different turns or to separate session and knowledge to feed input texts into the encoder, predict the response from the decoder.

```python
from pytorch_transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')    

sep = ' [SEP] '
context = [
    'So how did I do on my driving test ?', 
    'Do you want the truth ?', 
    'Of course , I do .' 
]
response = 'Well , you really did not do all that well .'

tokenized_context = sep.join([' '.join(tokenizer.tokenize(sent)) for sent in context])
tokenized_response = tokenizer.tokenize(response)

fin = 'train.src'
fout = 'train.tgt'
with open(fin, 'w', encoding='utf-8') as f:
    f.write(tokenized_context + '\n')
with open(fin, 'w', encoding='utf-8') as f:
    f.write(tokenized_response + '\n')
```

#### Binirization

```shell
PROJECT_PATH=/remote-home/wchen/project/DialogVED

USER_DIR=${PROJECT_PATH}/src
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

DATA_DIR=YourDatasetDir
PROCESSED_DIR=${DATA_DIR}/processed    # put train.src, train.tgt, dev.src, dev.tgt, test.src, test.tgt here
BINARY_DIR=${DATA_DIR}/binary          # binarized dir
TASK=translation_prophetnet            # 

fairseq-preprocess \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}
```

#### Train

Note: If your device does not support float16, remove --fp16.  If the GPU memory of your device is small and cannot support the default batch size, please remember to reduce the learning rate appropriately, or it will not converge normally. 

```shell
PRETRAINED_MODEL_PATH='/remote-home/wchen/models/dialogved_large.pt'

PROJECT_PATH='/remote-home/wchen/project/DialogVED'
ARCH=ngram_transformer_prophet_vae_large
NUM_WORKERS=10
CRITERION=ved_loss
TASK=translation_prophetnet
USER_DIR=${PROJECT_PATH}/src
DATA_DIR=YourDatasetDir
SAVE_DIR=${DATA_DIR}/checkpoints
TB_LOGDIR=${DATA_DIR}/tensorboard


fairseq-train \
  ${DATA_DIR}/binary \
  --fp16 \
  --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
  --lr 0.0003 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
  --criterion $CRITERION --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 4500 --max-sentences 16 \
  --num-workers ${NUM_WORKERS}  \
  --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.0 --weight-decay 0.01 \
  --encoder-layer-drop 0.0 \
  --save-dir ${SAVE_DIR} \
  --max-epoch 10 \
  --keep-last-epochs 10 \
  --max-source-positions 512 \
  --max-target-positions 128 \
  --kl-loss-weight 1.0 \
  --target-kl 5.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 1.0 \
  --masked-lm-loss-weight 0.0 \
  --tensorboard-logdir ${TB_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
```

#### Inference

Inference with fairseq-generate to generate targets for given processed test files.

```shell
BEAM=5
LENPEN=1.0
DATA_DIR=YourDatasetDir
CHECK_POINT=${SAVE_DIR}/checkpoint8.pt
OUTPUT_FILE=${DATA_DIR}/output.txt
PRED_FILE=${DATA_DIR}/pred.txt  # this the final prediction results
TASK=translation_prophetnet

fairseq-generate "${DATA_DIR}"/binary \
  --path "${CHECK_POINT}" \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --batch-size 64 \
  --gen-subset test \     
  --beam ${BEAM} \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  2>&1 >"${OUTPUT_FILE}"
grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > "${PRED_FILE}"
```

### Fine-tuning on DailyDialog, PersonaChat and DSTC7AVSD

#### Data preparation

We finetune DialogVED on three datasets [DailyDialog](https://arxiv.org/abs/1710.03957), [PersonaChat](https://arxiv.org/abs/1801.07243) and [DSTC7AVSD](https://arxiv.org/abs/1806.00525). You can download them according to the instructions in [PLATO](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO), or run our script as follows.

```shell
bash preprocess/get_data.sh
```

#### Preprocess

```shell
bash preprocess/process.sh
```

#### Binarization

```shell
bash preprocess/binarize.sh
```

#### Training

the script `train.sh` has three parameters, namely `p`, `t` and `d`.

- `p`: pretrained model **p**ath
- `t`: pretrained model **t**ype (`dialogved_standard`, `dialogved_large` or `dialogved_seq2seq`)
- `d`: fine-tuned **d**ataset (`dailydialog`, `personachat` or `dstc7avsd`)

**Note**: According to the feedback of some developers, if the GPU memory of your device is small and cannot support the default batch size, please reduce the learning rate appropriately, or it will not converge normally. 

```shell
bash train.sh -p /remote-home/models/dialogved_standard.pt -t dialogved_standard -d dailydialog
```

#### Inference

the script `infer.sh` has two parameters, namely `d` and `s`.

- `d`: fine-tuned **d**ataset (`dailydialog`, `personachat` or `dstc7avsd`)
- `s`: decoding **s**trategy (`greedy`, `beam` or `sampling`)

```shell
bash infer.sh -d dailydialog -s beam
```

#### Evaluation

the script `eval.sh` has one parameter, namely `d`.

- `d`: fine-tuned **d**ataset (`dailydialog`, `personachat` or `dstc7avsd`)

```shell
bash eval.sh -d dailydialog
```

### Reddit dataset for pre-training

The original Reddit data for pre-training has been shared on Baidu's online disk.

Link: https://pan.baidu.com/s/1--K9DiPtsSStV7yKQPyc7A
Extraction Code: 8grj

### How to Cite

If you extend or use this work, please cite the [paper](https://aclanthology.org/2022.acl-long.333/) where it was introduced:

```text
@inproceedings{chen-etal-2022-dialogved,
    title = "{DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation",
    author = "Chen, Wei and Gong, Yeyun and Wang, Song and Yao, Bolun and Qi, Weizhen and Wei, Zhongyu and Hu, Xiaowu and Zhou, Bartuer and Mao, Yi and Chen, Weizhu and Cheng, Biao and Duan, Nan",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.333",
    doi = "10.18653/v1/2022.acl-long.333",
    pages = "4852--4864",
    abstract = "Dialog response generation in open domain is an important research topic where the main challenge is to generate relevant and diverse responses. In this paper, we propose a new dialog pre-training framework called DialogVED, which introduces continuous latent variables into the enhanced encoder-decoder pre-training framework to increase the relevance and diversity of responses. With the help of a large dialog corpus (Reddit), we pre-train the model using the following 4 tasks, used in training language models (LMs) and Variational Autoencoders (VAEs) literature: 1) masked language model; 2) response generation; 3) bag-of-words prediction; and 4) KL divergence reduction. We also add additional parameters to model the turn structure in dialogs to improve the performance of the pre-trained model. We conduct experiments on PersonaChat, DailyDialog, and DSTC7-AVSD benchmarks for response generation. Experimental results show that our model achieves the new state-of-the-art results on all these datasets.",
}
```

