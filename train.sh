while getopts ":p:t:d:" opt
do
    case $opt in
        p)
          PRETRAINED_MODEL_PATH="$OPTARG"
        ;;
        t)
        PRETRAINED_MODEL_TYPE="$OPTARG"
        ;;
        d)
        DATASET="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

PROJECT_PATH='.'

if [ "$PRETRAINED_MODEL_TYPE" == "dialogved_standard" ]; then
  echo '-------- model type: dialogved standard --------'
  ARCH=ngram_transformer_prophet_vae_standard
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_large" ]; then
  echo '-------- model type: dialogved large --------'
  ARCH=ngram_transformer_prophet_vae_large
elif [ "$PRETRAINED_MODEL_TYPE" == "dialogved_seq2seq"  ]; then
  echo '-------- model type: dialogved seq2seq --------'
  ARCH=ngram_transformer_prophet_seq2seq
else
  echo 'model type '"$PRETRAINED_MODEL_TYPE"' not found!'
  exit 1
fi

if [ "$DATASET" == "dailydialog" ]; then
  echo '-------- fine-tune on dataset: dailydialog --------'
  NUM_WORKERS=10
  CRITERION=ved_loss
  TASK=ved_translate
  USER_DIR=${PROJECT_PATH}/src
  DATA_DIR=${PROJECT_PATH}/data/finetune/dailydialog
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
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "dstc7avsd" ]; then
  echo '-------- fine-tune on dataset: dstc7avsd --------'
  NUM_WORKERS=10
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
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
elif [ "$DATASET" == "personachat"  ]; then
  echo '-------- fine-tune on dataset: personachat --------'
  NUM_WORKERS=10
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
    --load-from-pretrained-model "${PRETRAINED_MODEL_PATH}"
else
#  echo 'dataset not found!'
  echo 'dataset '"$DATASET"' not found!'
  exit 1
fi
