# inherit from cuda (uses ubuntu18.04)
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# install dependencies
RUN apt update && apt install -y \
     software-properties-common \
     vim \
     git \
     make \
     htop \
     curl

# add python3.8 ppa
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt install -y python3.8 python3.8-venv

# create data Directory and set permissions
RUN mkdir -p /root/ssr/dev
RUN mkdir -p /root/ssr/data
WORKDIR /root/ssr/

# copy files to setup venv
COPY ./requirements.txt /root/ssr/dev/requirements.txt
COPY ./Makefile /root/ssr/dev/Makefile

# setup venv and install requirements
RUN /bin/bash -c "cd /root/ssr/dev/ && make init"

# copies files
COPY . /root/ssr/dev

CMD ["sh", "-c", "tail -f /dev/null"]
