# DialogVED
Code and released pre-trained model for our ACL 2022 paper: [DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation](https://arxiv.org/abs/2204.13031)

### News

- Pretrained checkpoints of DialogVED have been released!

### TODO

- ~~Release of pretrained models~~
- Requirements for running this code
- Detailed instructions for running the code

### Requirements

- torch==1.3.0
- fairseq==v0.9.0
- tensorboardX==1.7

### Pre-trained Models

We have released the following checkpoints for pre-trained models as described in the paper of [DialogVED](https://arxiv.org/abs/2204.13031). Download the pre-trained checkpoint and set the `load-from-pretrained-model` parameter in the fine-tuning running command.  

- [DialogVED-VAE-Standard](https://drive.google.com/file/d/1GLMrNAc2YEPJ-eiRcbFHGP0XGzAwKikM/view?usp=sharing)
- [DialogVED-VAE-Large](https://drive.google.com/file/d/1EucujAl8vXyrEDAyAb0SeLovzIX2_9tn/view?usp=sharing)
- [DialogVED-Seq2Seq](https://drive.google.com/file/d/1xiRMBPeaIUvKFbnKrf7etXPVyU1C1x56/view?usp=sharing)

**Note**: DialogVED-VAE-Standard has a size of latent size 32, where DialogVED-VAE-Large has a size of latent size 64. DialogVED-Seq2Seq has not latent variable, it's a pure seq2seq model with the same training setting like DialogVED. It may perform better in scenarios where diversity of responses is less important.   

### Fine-tuning

#### Data preparation

We finetune DialogVED on three datasets [DailyDialog](https://arxiv.org/abs/1710.03957), [PersonaChat](https://arxiv.org/abs/1801.07243) and [DSTC7AVSD](https://arxiv.org/abs/1806.00525). You can download them according to the instructions in [PLATO](https://github.com/PaddlePaddle/Research/tree/master/NLP/Dialogue-PLATO), or run our script as follows.

```shell
bash src/finetune/get_data.sh
```

#### DailyDialog

```shell
bash jobs/dailydialog/job.sh
```

#### PersonaChat

```shell
bash jobs/personachat/job.sh
```

#### DSTC7AVSD

```shell
bash jobs/dstc7avsd/job.sh
```

### Pre-training

#### Get Reddit dataset

We used a script provided from [DialoGPT](https://github.com/microsoft/DialoGPT) to get the latest Reddit dataset. Follow the instructions of https://github.com/microsoft/DialoGPT and run `python demo.py --data full`, you will download a raw reddit dataset called `train.tsv`.

The raw data is large, and it may take several days to download it. We provide a tiny sample [here]() for testing.

#### Run Pre-training

```shell
bash pretrain/reddit/job_vae.sh
```
