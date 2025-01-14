#!/usr/bin/env bash

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

echo "Container nvidia build = " $NVIDIA_BUILD_ID

electra_model=${1:-"google/electra-base-discriminator"}
epochs=${2:-"2"}
batch_size=${3:-"16"}
infer_batch_size=${4:-"$batch_size"}
learning_rate=${5:-"4e-4"}
precision=${6:-"amp"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
SQUAD_VERSION=${9:-"1.1"}
squad_dir=${10:-"/workspace/electra/data/download/squad/v$SQUAD_VERSION"}
OUT_DIR=${11:-"results/"}
mode=${12:-"train_eval"}
env=${13:-"interactive"}
cache_dir=${14:-"$squad_dir"}
max_steps=${15:-"-1"}

#bash scripts/run_squad.sh google/electra-base-discriminator 2 16 64 4e-4 amp 8 1 1.1 /workspace/data/squad/v1.1/ saved_models/output_dir  train_eval interactive saved_models/cache_dir

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "amp" ] ; then
  echo "mixed-precision training activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command="python "
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command="python -m torch.distributed.launch --nproc_per_node=$num_gpu "
fi

if [ "$env" = "cluster" ] ; then
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" "
fi

v2=""
echo "Running SQuAD-v$SQUAD_VERSION"
if [ "$SQUAD_VERSION" = "2.0" ] ; then
  v2=" --version_2_with_negative "
fi

CMD=" $mpi_command run_squad.py "
if [ "$mode" = "train" ] ; then
  CMD+=" --do_train "
  CMD+=" --per_gpu_train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+=" --per_gpu_eval_batch_size=$infer_batch_size "
  CMD+=" --do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+=" --do_eval "
  CMD+=" --per_gpu_eval_batch_size=$infer_batch_size "
else
  CMD+=" --do_train "
  CMD+=" --per_gpu_train_batch_size=$batch_size "
  CMD+=" --per_gpu_eval_batch_size=$infer_batch_size "
  CMD+=" --do_eval "
fi

CMD+=" $v2 "
CMD+=" --data_dir $squad_dir "
CMD+=" --do_lower_case "
CMD+=" --model_name_or_path=$electra_model "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion 0.05 "
CMD+=" --weight_decay 0.01 "
CMD+=" --layerwise_lr_decay 0.8 "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --beam_size 5 "
CMD+=" --null_score_diff_threshold -5.6 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" $use_fp16"
CMD+=" --cache_dir=$cache_dir "
CMD+=" --max_steps=$max_steps "
CMD+=" --overwrite_output_dir "
CMD+=" --vocab_file=/workspace/electra/vocab/vocab.txt "

LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
