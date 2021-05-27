#!/bin/bash

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run --gpus all -it --rm --ipc=host \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  -v /mnt/shared/yuya/transformers:/workspace/electra \
  -v /mnt/dldata/yuya/electra_tf_data/:/workspace/electra/pretraining_data \
  -v /mnt/shared/yuya/dldata/electra_pyt:/workspace/electra/saved_models \
  -v /mnt/dldata/bert_tf2/download:/workspace/data electra:pyt
