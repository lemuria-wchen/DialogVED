# mode can be either CLUSTER or STANDALONE
MODE=CLUSTER
#MODE=STANDALONE

if [ "${MODE}" = CLUSTER ]
then PREFIX=/mnt/dialogue
else PREFIX=/home/v-wchen2/data/dialogue
fi

#################################################################################################
# training
SUFFIX='_drop_knowledge'
BASE_DIR=${PREFIX}/finetune/personachat
SAVE_DIR=${BASE_DIR}/checkpoints/${SUFFIX}
TENSORBOARD_LOGDIR=${BASE_DIR}/tensorboard/${SUFFIX}

# set pretrained model path
PRETRAINED_MODEL=${PREFIX}/checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

# parameters that do not require additional parameters
USER_DIR=./prophetnet_dialog
ARCH=ngram_transformer_prophet_large
CRITERION=ngram_language_loss

fairseq-train \
  --fp16 \
  --user-dir ${USER_DIR} --task translation_prophetnet --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.999)' --clip-norm 0.1 \
  --lr 0.0001 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 1000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --criterion ${CRITERION} --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 3000 --max-sentences 6 \
  --num-workers 1 \
  --load-from-pretrained-model ${PRETRAINED_MODEL} \
  --load-sep \
  --ddp-backend=no_c10d --max-epoch 5 \
  --max-source-positions 512 --max-target-positions 128 \
  --skip-invalid-size-inputs-valid-test \
  --seed 1 \
  --save-dir ${SAVE_DIR} \
  --keep-last-epochs 5 \
  --tensorboard-logdir ${TENSORBOARD_LOGDIR} \
  ${BASE_DIR}/binary

#################################################################################################
# inference
BEAM=5
LENPEN=1
NAME=_pelt${LENPEN}_beam${BEAM}${SUFFIX}
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
OUTPUT_FILE=${BASE_DIR}/output${NAME}.txt

fairseq-generate ${BASE_DIR}/binary \
  --path ${CHECK_POINT} \
  --user-dir ./prophetnet \
  --task translation_prophetnet \
  --batch-size 64 \
  --gen-subset test \
  --beam ${BEAM} \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  2>&1 >${OUTPUT_FILE}

grep ^H ${OUTPUT_FILE} | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${BASE_DIR}/sort${NAME}.txt

#################################################################################################
# evaluation
HYP_NAME=sort${NAME}.txt
OUT_NAME=result${SUFFIX}.txt

if [ -f ${BASE_DIR}/${OUT_NAME} ]; then
  rm ${BASE_DIR}/${OUT_NAME}
fi

python utils/evaluate.py \
  -name personachat \
  -hyp ${BASE_DIR}/${HYP_NAME} \
  -ref ${BASE_DIR}/processed/test.tgt \
  -know ${BASE_DIR}/original_data/dial.test \
  -out ${BASE_DIR}/${OUT_NAME}

# print env variables
echo ${SAVE_DIR}
echo ${TENSORBOARD_LOGDIR}
echo ${PRETRAINED_MODEL}
echo ${NAME}
echo ${OUTPUT_FILE}
echo ${HYP_NAME}
echo ${OUT_NAME}
