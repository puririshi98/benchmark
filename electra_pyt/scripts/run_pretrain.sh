python -m torch.distributed.launch --nproc_per_node=16 run_pretrain.py \
    --model_name='test02' \
    --data_dir='pretraining_data/' \
    --pretrain_hdf5='pretraining_data/pretrain_hdf5_seq_512/*.hdf5' \
    --num_train_steps=766000 \
    --num_warmup_steps=10000 \
    --disc_weight=50.0 \
    --generator_hidden_size=0.3333333 \
    --learning_rate=2e-4 \
    --train_batch_size=16 \
    --max_seq_length=512 \
    --save_checkpoints_steps=100 \
    --keep_checkpoint_max=3 \
    --gradient_accumulation_steps=1 \
    --restore_checkpoint='latest'