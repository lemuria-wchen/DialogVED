# mode can be either CLUSTER or STANDALONE or LOCAL
MODE=STANDALONE

if [ "${MODE}" = CLUSTER ]
then
  PREFIX=/mnt/dialogue
elif [ "${MODE}" = STANDALONE ]
then
  PREFIX=/home/v-wchen2/data/dialogue
else
  PREFIX=/mnt/d/dialogue
fi

PREFIX='/home/v-bolunyao/wchen2'

# training
########################################################################################################################
SUFFIX='_released_vae'

DATA_DIR=${PREFIX}/finetune/dailydialog

BINARY_DIR=${DATA_DIR}/binary
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/DialogVED

# use the latest seq2seq checkpoints
#PRETRAINED_MODEL=/home/v-wchen2/data/dialogue/checkpoints/checkpoint2.pt
PRETRAINED_MODEL=/mnt/my_outputs/wchen2/checkpoints/_seq2seq_lm_1800_16_no_pe/checkpoint2.pt

# parameters that do not require additional parameters
NUM_WORKERS=10
ARCH=ngram_transformer_prophet_vae_standard
CRITERION=ved_loss
TASK=ved_translate

echo 'seq2seq pretrain starting ...'

"$(which fairseq-train)" \
  ${BINARY_DIR} \
  --fp16 \
  --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
  --lr 0.0002 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
  --warmup-init-lr 1e-07 --warmup-updates 2000 \
  --criterion $CRITERION \
  --update-freq 8 --max-tokens 1500 --max-sentences 8 \
  --num-workers ${NUM_WORKERS}  \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0.0 \
  --weight-decay 0.01 \
  --encoder-layer-drop 0.0 \
  --save-dir ${SAVE_DIR} \
  --max-epoch 10 \
  --keep-last-epochs 10 \
  --max-source-positions 512 \
  --max-target-positions 128 \
  --target-kl 3.0 \
  --kl-loss-weight 1.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 1.0 \
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
  --add-cls-to-source \
  --load-from-pretrained-model ${PRETRAINED_MODEL}
