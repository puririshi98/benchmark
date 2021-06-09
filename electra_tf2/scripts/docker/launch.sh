#!/bin/bash

# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CMD=${1:-/bin/bash}
NV_VISIBLE_DEVICES=${2:-"all"}
DOCKER_BRIDGE=${3:-"host"}

docker run -it --rm \
  --gpus device=$NV_VISIBLE_DEVICES \
  --net=$DOCKER_BRIDGE \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  -v $PWD:/workspace/electra \
  -v /mnt/dldata/electra:/workspace/electra/data \
  -v /mnt/dldata/sharatht/electra_tf2_pretrained_checkpoints:/checkpoints_in \
  -v /mnt/dldata/sharatht/electra_checkpoints_gen_and_disc:/checkpoints_out \
  -v /mnt/dldata/sharatht/electra:/workspace/electra/checkpoints \
  electra $CMD
