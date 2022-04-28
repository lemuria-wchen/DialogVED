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
SUFFIX='_released_vae_standard'

DATA_DIR=${PREFIX}/pretrain/reddit
BINARY_DIR=${DATA_DIR}/binary/finetune_sample
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/DialogVED

# use the latest seq2seq checkpoints
PRETRAINED_MODEL=/home/v-wchen2/data/dialogue/checkpoints/checkpoint2.pt

# parameters that do not require additional parameters
NUM_WORKERS=10
ARCH=ngram_transformer_prophet_vae_standard
CRITERION=ved_loss
TASK=ved_translate

echo 'vae pretrain starting ...'

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
  --add-cls-to-source \
  --target-kl 5.0 \
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
  --load-from-pretrained-model ${PRETRAINED_MODEL}


# interactive
CHECK_POINT=${SAVE_DIR}/checkpoint_last.pt

"$(which fairseq-interactive)" \
  ${DATA_DIR}/binary \
  --path ${CHECK_POINT} \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --model-overrides "{'deterministic':'True', 'mask_source': 'False'}" \
  --no-repeat-ngram-size 3 \
  --source-lang src \
  --target-lang tgt \
  --beam 1
