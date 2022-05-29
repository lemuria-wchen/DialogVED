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
DATA_DIR=${PROJECT_PATH}/data/finetune/dstc7avsd
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
  --update-freq 4 --max-tokens 1500 --max-sentences 8 \
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


BEAM_SIZE=1
#SUFFIX='_vae_kl_5_standard_beam_1'
#SUFFIX='_vae_kl_5_standard_beam_5'
SUFFIX='_vae_kl_5_large_beam_1'
#CHECK_POINT=${SAVE_DIR}/checkpoint_best.pt
CHECK_POINT=${SAVE_DIR}/checkpoint4.pt
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
  -name dstc7avsd \
  -hyp ${SORTED_OUTPUT_FILE} \
  -ref ${DATA_DIR}/processed/test_multi_refs.tgt \
  -out ${EVALUATE_LOG_PATH}


# sampling
#SUFFIX='_vae_kl_5_large_beam_5'
#SUFFIX='_vae_kl_5_standard_beam_5'
#SUFFIX='_vae_kl_5_standard_sampling_100'
SUFFIX='_vae_kl_5_standard_sampling_100'
CHECK_POINT=${SAVE_DIR}/checkpoint4.pt
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
  --sampling \
  --sampling-topk 5 \
  --nbest 1 \
  --beam 1 \
  2>&1 >"${UNSORTED_OUTPUT_FILE}"


grep ^H "${UNSORTED_OUTPUT_FILE}" | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > ${SORTED_OUTPUT_FILE}
EVALUATE_LOG_PATH=${DATA_DIR}/result${SUFFIX}.txt

python utils/evaluate.py \
  -name dstc7avsd \
  -hyp ${SORTED_OUTPUT_FILE} \
  -ref ${DATA_DIR}/processed/test_multi_refs.tgt \
  -out ${EVALUATE_LOG_PATH}

