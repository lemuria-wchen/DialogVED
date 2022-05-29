# this script is used pre-process pre-train dialogue corpus
# in this paper we consider two corpus, i.e., reddit, twitter
# note that twitter corpus is not friendly enough and has been discarded
# running this script will cause all files with the same name in the `BINARY_DIR` folder will be overwritten
# running this script could take hours, depends on the performance of machine processor

PREFIX=/home/v-wchen2/data/dialogue

PROJECT_PATH='.'
USER_DIR=${PROJECT_PATH}/prophetnet_dialog
VOCAB_PATH=${PROJECT_PATH}/vocab.txt
NUM_WORKERS=20

# pre-process (single shard)
# this command will write more than ten gigabytes of binaries
########################################################################################################################
DATA_DIR=${PREFIX}/pretrain/reddit
PROCESSED_DIR=${DATA_DIR}/processed/finetune
BINARY_DIR=${DATA_DIR}/binary/finetune

"$(which fairseq-preprocess)" \
  --fp16 \
  --user-dir ${USER_DIR} \
  --task seq2seq_vae \
  --trainpref ${PROCESSED_DIR}/train.src \
  --validpref ${PROCESSED_DIR}/valid.src \
  --destdir ${BINARY_DIR} \
  --srcdict ${VOCAB_PATH} \
  --tgtdict ${VOCAB_PATH} \
  --workers ${NUM_WORKERS}


# pre-process (multiple shards)
# this command will write ten binary files of about one gigabyte
# this command is recommended for machines with limit memory
########################################################################################################################
for ((i = 1; i <= 10; i++))
do
  PROCESSED_DIR=${DATA_DIR}/processed/shards
  BINARY_DIR=${DATA_DIR}/binary/shards/part"${i}"

  "$(which fairseq-preprocess)" \
    --fp16 \
    --user-dir ${USER_DIR} \
    --task seq2seq_vae \
    --source-lang src \
    --target-lang tgt \
    --trainpref ${PROCESSED_DIR}/train.part"${i}" \
    --validpref ${PROCESSED_DIR}/valid.part"${i}" \
    --destdir ${BINARY_DIR} \
    --srcdict ${VOCAB_PATH} \
    --tgtdict ${VOCAB_PATH} \
    --workers ${NUM_WORKERS}
done
