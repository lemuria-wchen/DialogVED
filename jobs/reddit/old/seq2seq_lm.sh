# mode can be either CLUSTER or STANDALONE or LOCAL
MODE=CLUSTER

if [ "${MODE}" = CLUSTER ]
then
  PREFIX=/mnt/dialogue
elif [ "${MODE}" = STANDALONE ]
then
  PREFIX=/home/v-wchen2/data/dialogue
else
  PREFIX=/mnt/d/dialogue
fi

# training
########################################################################################################################
SUFFIX='_seq2seq_lm_1800_16'

DATA_DIR=${PREFIX}/pretrain/reddit
BINARY_DIR=${DATA_DIR}/binary/finetune
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/prophetnet_dialog
PRETRAINED_MODEL=${DATA_DIR}/checkpoints_pretrain_mask1/checkpoint19.pt

# parameters that do not require additional parameters
NUM_WORKERS=10
ARCH=ngram_transformer_prophet_seq2seq
CRITERION=ngram_language_loss
TASK=seq2seq_vae

echo 'pretrain starting ...'

"$(which fairseq-train)" \
  ${BINARY_DIR} \
  --fp16 \
  --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
  --lr 0.0002 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
  --warmup-init-lr 1e-07 --warmup-updates 2000 \
  --criterion $CRITERION \
  --update-freq 8 --max-tokens 1800 --max-sentences 16 \
  --num-workers ${NUM_WORKERS}  \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0.0 \
  --weight-decay 0.01 \
  --encoder-layer-drop 0.0 \
  --save-dir ${SAVE_DIR} \
  --max-epoch 10 \
  --keep-last-epochs 10 \
  --max-source-positions 128 \
  --max-target-positions 64 \
  --kl-loss-weight 0.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 0.0 \
  --masked-lm-loss-weight 1.0 \
  --log-interval 1000 \
  --save-interval-updates 10000 \
  --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --load-from-pretrained-model ${PRETRAINED_MODEL}

