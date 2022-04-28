# DialogVED
Code and released pre-trained model for our ACL 2022 paper: "DialogVED: A Pre-trained Latent Variable Encoder-Decoder Model for Dialog Response Generation"

### TODO

- Release of pretrained models
- Requirements for running this code
- Detailed instructions for running the code

### Pre-training

```shell
bash pretrain/reddit/job_vae.sh
```

### Fine-tuning

#### for w

```shell
bash jobs/dailydialog/job.sh
```

```shell
bash jobs/personachat/job.sh
```

```shell
bash jobs/dstc7avsd/job.sh
```
