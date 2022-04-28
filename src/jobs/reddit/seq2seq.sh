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

# training
########################################################################################################################
SUFFIX='_released_seq2seq'

DATA_DIR=${PREFIX}/pretrain/reddit
BINARY_DIR=${DATA_DIR}/binary/finetune_sample
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/DialogVED

# use the latest seq2seq checkpoints
PRETRAINED_MODEL=/home/v-wchen2/data/dialogue/checkpoints/checkpoint2.pt
#PRETRAINED_MODEL=${PREFIX}/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt


# parameters that do not require additional parameters
NUM_WORKERS=10
ARCH=ngram_transformer_prophet_seq2seq
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
  --mask-source \
  --masked-prob 0.15 \
  --masked-span-len 2 \
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


## interactive
#CHECK_POINT=${SAVE_DIR}/checkpoint_last.pt
#
#"$(which fairseq-interactive)" \
#  ${DATA_DIR}/binary \
#  --path ${CHECK_POINT} \
#  --user-dir ${USER_DIR} \
#  --task ${TASK} \
#  --model-overrides "'mask_source': 'False'}" \
#  --no-repeat-ngram-size 3 \
#  --source-lang src \
#  --target-lang tgt \
#  --beam 1

