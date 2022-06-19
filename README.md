# DialogVED

Code and released pre-trained model for our ACL 2022 paper: [DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation](https://aclanthology.org/2022.acl-long.333/).

### News

- Fixed bugs in dailydialog, updated new training and evaluation scripts. (2022.06.19)
- Optimize code structure and remove redundant code. (2022.05.29) 
- Pretrained checkpoints of DialogVED have been released! (2022.05.17)

### TODO

- Pre-trained scripts are scheduled to be released.

### Requirements

- python==3.7
- torch==1.3.0
- fairseq==0.9.0
- tensorboardX==1.7
- pytorch_transformers
- sklearn

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

**Note**: DialogVED-VAE-Standard has a size of latent size 32, where DialogVED-VAE-Large has a size of latent size 64. DialogVED-Seq2Seq has not latent variable, it's a pure seq2seq model with the same training setting like DialogVED. It may perform better in scenarios where diversity of responses is less important.   

### Fine-tuning

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
- `t`: pretrained model **t**ype (`dialogved_standard`, `dialogved_large` or `dialogved_seq`)
- `d`: fine-tuned **d**ataset (`dailydialog`, `personachat` or `dstc7avsd`)

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

### How to Cite

If you extend or use this work, please cite the [paper](https://aclanthology.org/2022.acl-long.333/) where it was introduced:

```text
@inproceedings{chen-etal-2022-dialogved,
    title = "{D}ialog{VED}: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation",
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


[//]: # (#### Fine-tuning on DailyDialog)

[//]: # ()
[//]: # (```shell)

[//]: # (bash jobs/dailydialog/job.sh)

[//]: # (```)

[//]: # ()
[//]: # (#### Fine-tuning on PersonaChat)

[//]: # ()
[//]: # (```shell)

[//]: # (bash jobs/personachat/job.sh)

[//]: # (```)

[//]: # ()
[//]: # (#### Fine-tuning on DSTC7AVSD)

[//]: # ()
[//]: # (```shell)

[//]: # (bash jobs/dstc7avsd/job.sh)

[//]: # (```)

[//]: # ()
[//]: # (### Pre-training)

[//]: # ()
[//]: # (#### Get Reddit dataset)

[//]: # ()
[//]: # (We used a script provided from [DialoGPT]&#40;https://github.com/microsoft/DialoGPT&#41; to get the latest Reddit dataset. Follow the instructions of https://github.com/microsoft/DialoGPT and run `python demo.py --data full`, you will download a raw reddit dataset called `train.tsv`.)

[//]: # ()
[//]: # (The raw data is large, and it may take several days to download it. We provide a tiny sample [here]&#40;&#41; for testing.)

[//]: # ()
[//]: # (#### Run Pre-training)

[//]: # ()
[//]: # (```shell)

[//]: # (bash jobs/reddit/vae_standard.sh)

[//]: # (```)
