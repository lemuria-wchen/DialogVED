while getopts ":d:" opt
do
    case $opt in
        d)
        DATASET="$OPTARG"
        ;;
        ?)
        echo "未知参数"
        exit 1;;
    esac
done

echo '-------- evaluate on dataset: '"$DATASET"'--------'

PROJECT_PATH='.'
DATA_DIR=${PROJECT_PATH}/data/finetune/${DATASET}
PRED_FILE=${DATA_DIR}/pred.txt

if [ "$DATASET" == "dailydialog" ]; then
  python utils/evaluate.py \
    -name dailydialog \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/test.tgt
elif [ "$DATASET" == "dstc7avsd" ]; then
  python utils/evaluate.py \
    -name dstc7avsd \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/test_multi_refs.tgt
elif [ "$DATASET" == "personachat"  ]; then
  python utils/evaluate.py \
    -name personachat \
    -hyp "${PRED_FILE}" \
    -ref "${DATA_DIR}"/processed/test.tgt
else
  echo 'dataset '"$DATASET"' not found!'
  exit 1
fi
