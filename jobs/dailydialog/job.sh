# training
########################################################################################################################
PROJECT_PATH=/remote-home/wchen/project/DialogVED

# pretrained model path
PRETRAINED_MODEL=PROJECT_PATH/dialogved_standard.pt

NUM_WORKERS=10
ARCH=ngram_transformer_prophet_vae_standard
CRITERION=ved_loss
TASK=ved_translate

USER_DIR=${PROJECT_PATH}/src
DATA_DIR=${PROJECT_PATH}/data/finetune/dailydialog
SAVE_DIR=${DATA_DIR}/checkpoints
TB_LOGDIR=${DATA_DIR}/tensorboard

"$(which fairseq-train)" \
  ${DATA_DIR}/binary \
   --fp16 \
  --user-dir ${USER_DIR} --task ${TASK} --arch ${ARCH} \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 1.0 \
  --lr 0.0003 --lr-scheduler inverse_sqrt --min-lr 1e-09 \
  --warmup-init-lr 1e-07 --warmup-updates 2000 \
  --criterion $CRITERION --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 1500 --max-sentences 8 \
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
  --target-kl 5.0 \
  --kl-loss-weight 1.0 \
  --cls-bow-loss-weight 0.0 \
  --latent-bow-loss-weight 1.0 \
  --masked-lm-loss-weight 0.0 \
  --log-interval 1000 \
  --save-interval-updates 10000 \
  --tensorboard-logdir ${TB_LOGDIR} \
  --dataset-impl mmap \
  --empty-cache-freq 64 \
  --seed 1 \
  --skip-invalid-size-inputs-valid-test \
  --distributed-no-spawn \
  --ddp-backend no_c10d \
  --add-cls-to-source \
  --load-from-pretrained-model ${PRETRAINED_MODEL}

#################################################################################################
# inference
BEAM=5
LENPEN=1
CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
OUTPUT_FILE=${DATA_DIR}/output.txt
PRED_FILE=${DATA_DIR}/pred.txt

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

grep ^H ${OUTPUT_FILE} | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${PRED_FILE}

#################################################################################################
# evaluation

python utils/evaluate.py \
  -name dailydialog \
  -hyp ${PRED_FILE} \
  -ref ${DATA_DIR}/processed/test.tgt
