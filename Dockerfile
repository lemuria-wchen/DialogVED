FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

# Labels for the docker
LABEL description="This docker has pytorch 1.3, cuda 10.1, apex, meteor, nltk, tensorboardX and etc."

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex&& pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..

RUN apt install default-jdk
RUN curl https://install.meteor.com/ | sh
RUN pip install tensorboard tensorboardX pycocoevalcap nltk argparse numpy
RUN git clone https://github.com/pytorch/fairseq && cd fairseq && git checkout tags/v0.9.0 && pip install --editable . && cd ..

# login
# docker login -u dandanlemuria

# build docker image based on dockerfile
# docker build -t dandanlemuria/dialogue_pretrain:1.0 . --shm-size=4g

# push to dockerhub
# docker push dandanlemuria/dialogue_pretain:1.0
