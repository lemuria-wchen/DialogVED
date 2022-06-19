# training
########################################################################################################################
PROJECT_PATH=/remote-home/wchen/project/DialogVED

# pretrained model path
PRETRAINED_MODEL=${PROJECT_PATH}/dialogved_standard.pt

NUM_WORKERS=10
ARCH=ngram_transformer_prophet_vae_large
CRITERION=ved_loss
TASK=ved_translate

USER_DIR=${PROJECT_PATH}/src
DATA_DIR=${PROJECT_PATH}/data/finetune/personachat
SAVE_DIR=${DATA_DIR}/checkpoints
TB_LOGDIR=${DATA_DIR}/tensorboard

fairseq-train \
  ${DATA_DIR}/binary \
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
  --kl-loss-weight 1.0 \
  --target-kl 5.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 1.0 \
  --masked-lm-loss-weight 0.0 \
  --tensorboard-logdir ${TB_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --load-from-pretrained-model ${PRETRAINED_MODEL}

#################################################################################################
# inference
BEAM=5
LENPEN=1
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
OUTPUT_FILE=${DATA_DIR}/output.txt
PRED_FILE=${DATA_DIR}/pred.txt

fairseq-generate \
  ${DATA_DIR}/binary \
  --path ${CHECK_POINT} \
  --user-dir ${USER_DIR} \
  --task ${TASK} \
  --batch-size 64 \
  --gen-subset test \
  --num-workers 4 \
  --no-repeat-ngram-size 3 \
  --lenpen ${LENPEN} \
  --beam ${BEAM} \
  2>&1 >"${OUTPUT_FILE}"

#################################################################################################
# evaluation
grep ^H "${OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${PRED_FILE}

python utils/evaluate.py \
  -name personachat \
  -hyp ${PRED_FILE} \
  -ref ${DATA_DIR}/processed/test.tgt
