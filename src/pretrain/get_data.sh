# download twitter & reddit dataset for dialogue system pre-training

# twitter
# output_file_path=/home/v-wchen2/Data/dialogue/pretrain
output_file_path=/mnt/d/dialogue/pretrain

if [ ! -d ${output_file_path}/chat_corpus ]; then
  mkdir -p ${output_file_path}/chat_corpus
fi

echo 'downloading twitter data'

git clone https://github.com/marsan-ma/chat_corpus.git ${output_file_path}/chat_corpus

cur_dir=$(pwd)
cd ${output_file_path}/chat_corpus && cat twitter_en_big.txt.gz.part* > twitter_en_big.txt.gz

if [ ! -d ${output_file_path}/twitter/original_data ]; then
  mkdir -p ${output_file_path}/twitter/original_data
fi

gzip -d twitter_en_big.txt.gz
mv twitter_en_big.txt ../twitter/original_data
cd .. && rm -rf chat_corpus && cd "${cur_dir}" || return

# reddit
