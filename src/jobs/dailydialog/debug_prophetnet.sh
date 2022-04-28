PREFIX=/home/v-bolunyao
DATA_DIR=${PREFIX}/finetune/dailydialog

PROJECT_PATH='/home/v-bolunyao/repo/DialogVED/src'
USER_DIR=${PROJECT_PATH}/prophetnet
NUM_WORKERS=20

SUFFIX='_seq2seq_ck5_prophetnet'

SAVE_DIR=${DATA_DIR}/checkpoints/${SUFFIX}
TENSORBOARD_LOGDIR=${DATA_DIR}/tensorboard/${SUFFIX}
BINARY_DIR=${DATA_DIR}/binary/

PRETRAINED_MODEL=/mnt/my_outputs/wchen2/checkpoints/_seq2seq_lm_1800_16/checkpoint5.pt

USER_DIR=./prophetnet
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss


fairseq-train \
  --fp16 \
  --user-dir ${USER_DIR} --task translation_prophetnet --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 1.0 \
  --lr 0.0003  \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --criterion ${CRITERION} --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 4500 --max-sentences 16 \
  --num-workers 1 \
  --load-from-pretrained-model ${PRETRAINED_MODEL} \
  --ddp-backend=no_c10d --max-epoch 5 \
  --max-source-positions 512 --max-target-positions 128 \
  --skip-invalid-size-inputs-valid-test \
  --seed 1 \
  --save-dir ${SAVE_DIR} \
  --keep-last-epochs 5 \
  --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
  ${DATA_DIR}/binary


BEAM=5
LENPEN=1
NAME=_pelt${LENPEN}_beam${BEAM}${SUFFIX}
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
OUTPUT_FILE=${DATA_DIR}/output${NAME}.txt

fairseq-generate ${DATA_DIR}/binary \
  --path ${CHECK_POINT} \
  --user-dir ${USER_DIR} \
  --task translation_prophetnet \
  --batch-size 64 \
  --gen-subset test \
  --beam ${BEAM} \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  2>&1 >${OUTPUT_FILE}

grep ^H ${OUTPUT_FILE} | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${DATA_DIR}/sort${NAME}.txt


HYP_NAME=sort${NAME}.txt
OUT_NAME=result${SUFFIX}.txt


python utils/evaluate.py \
  -name dailydialog \
  -hyp ${DATA_DIR}/${HYP_NAME} \
  -ref ${DATA_DIR}/processed/test.tgt \
  -out ${DATA_DIR}/${OUT_NAME}

