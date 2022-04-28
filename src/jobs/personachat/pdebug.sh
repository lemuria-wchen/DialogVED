PREFIX=/home/v-bolunyao

DATA_DIR=${PREFIX}/finetune/personachat

PROJECT_PATH='/home/v-bolunyao/repo/DialogVED/src'
USER_DIR=${PROJECT_PATH}/DialogVED
NUM_WORKERS=20

SUFFIX='_seq2seq_ck5_without_mask'
SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
BINARY_DIR=${DATA_DIR}/binary/

PRETRAINED_MODEL=/mnt/my_outputs/wchen2/checkpoints/_seq2seq_lm_1800_16/checkpoint6.pt

ARCH=ngram_transformer_prophet_seq2seq
CRITERION=ved_loss
TASK=translation_prophetnet


fairseq-train \
  ${BINARY_DIR} \
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
  --kl-loss-weight 0.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 0.0 \
  --masked-lm-loss-weight 0.0 \
  --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --load-from-pretrained-model ${PRETRAINED_MODEL}


BEAM_SIZE=5
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
TASK=translation_prophetnet

UNSORTED_OUTPUT_FILE=${DATA_DIR}/unsorted${SUFFIX}.txt
SORTED_OUTPUT_FILE=${DATA_DIR}/sorted${SUFFIX}.txt

fairseq-generate \
  ${DATA_DIR}/binary \
  --path ${CHECK_POINT} \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --batch-size 64 \
  --gen-subset test \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen 1 \
  --beam ${BEAM_SIZE} \
  2>&1 >"${UNSORTED_OUTPUT_FILE}"


grep ^H "${UNSORTED_OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${SORTED_OUTPUT_FILE}
EVALUATE_LOG_PATH=${DATA_DIR}/result${SUFFIX}.txt

python utils/evaluate.py \
  -name personachat \
  -hyp ${SORTED_OUTPUT_FILE} \
  -ref ${DATA_DIR}/processed/test.tgt \
  -know ${DATA_DIR}/original_data/dial.test \
  -out ${EVALUATE_LOG_PATH}
