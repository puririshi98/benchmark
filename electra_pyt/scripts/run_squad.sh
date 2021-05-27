#!/usr/bin/env bash
export SQUAD_VERSION=1.1
v2=""
echo "Running SQuAD-v$SQUAD_VERSION"
if [ "$SQUAD_VERSION" = "2.0" ] ; then
  v2=" --version_2_with_negative "
fi
python -m torch.distributed.launch --nproc_per_node=16 run_squad.py $v2 \
  --model_type electra \
  --model_name_or_path google/electra-base-discriminator \
  --cache_dir saved_models/cache_dir \
  --do_eval \
  --do_train \
  --do_lower_case \
  --seed 1213 \
  --train_file /workspace/data/squad/v$SQUAD_VERSION/train-v$SQUAD_VERSION.json \
  --predict_file /workspace/data/squad/v$SQUAD_VERSION/dev-v$SQUAD_VERSION.json \
  --threads 16 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 64 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --layerwise_lr_decay 0.8 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --warmup_steps 50 \
  --beam_size 4 \
  --null_score_diff_threshold -5.6 \
  --output_dir saved_models/output_dir \
  --logging_steps 200 \
  --save_steps 0 \
  --overwrite_output_dir \
  --fp16


