# mode can be either CLUSTER or STANDALONE or LOCAL
#MODE=CLUSTER
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

DATA_DIR=${PREFIX}/finetune/dailydialog/
PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/prophetnet_dialog
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

# step 1: pre-process
########################################################################################################################
PROCESSED_DIR=${DATA_DIR}/processed/
BINARY_DIR=${DATA_DIR}/binary/

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task seq2seq_vae \
  --source-lang src \
  --target-lang tgt \
  --trainpref ${PROCESSED_DIR}/train \
  --validpref ${PROCESSED_DIR}/valid \
  --testpref ${PROCESSED_DIR}/test \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}

# step 2: fine-tune -> continue training
########################################################################################################################
SUFFIX='_vae_bow_lm_standard_1800_16'
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
PRETRAINED_MODEL=${PREFIX}/checkpoints/_vae_bow_lm_standard_1800_16/checkpoint1.pt
ARCH=ngram_transformer_prophet_vae_standard
CRITERION=ngram_language_loss
TASK=seq2seq_vae

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
  --load-from-pretrained-model ${PRETRAINED_MODEL} \
  --add-cls-to-source \
  --masked-source

