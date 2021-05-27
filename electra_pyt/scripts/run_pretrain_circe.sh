#!/bin/bash
#SBATCH -A joc
#SBATCH -p batch
#SBATCH -N 8
#SBATCH -t 8:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --overcommit

set -eux

# The following variables variables need to be set
# Base container to be used
readonly docker_image="gitlab-master.nvidia.com/dl/joc/electra_pyt:pretrain"
readonly datadir="/gpfs/fs1/yuya/electra_pyt_data/"
# Location of dataset for phase 1
# Location of dataset for phase 2
#readonly datadir_phase2="/gpfs/fs1/sharatht/electra_tf2_data/pretrain_tfrecords"


readonly mounts=".:/workspace/electra,${datadir}:/workspace/electra/pretraining_data"

DGXSYSTEM=DGX2-H
cluster=""
if [[ "${DGXSYSTEM}" == DGX2* ]]; then
    cluster='circe'
fi
if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
    cluster='selene'
fi


#=======[ Set up multinode if applicable ]========#
SRUN_HOST=''
SRUN_CONT=''
if [[ -z "$SLURM_JOB_ID" ]]; then
  hosts=( `hostname` )
  export SLURM_NNODES=1
else
  hosts=( `scontrol show hostname |tr "\n" " "` )
  SRUN_HOST='srun --mem=0 -N 1 -n 1 -w $hostn'
  SRUN_CONT='srun --mem=0 -N 1 -n 1 -w $hostn --container-image="$CONTAINER" $MOUNTS'
fi
MASTER_IP="$(getent hosts "${hosts[0]}" | cut -d ' ' -f1 | head -n1)"
PORT=$((4242 + RANDOM%1000))

DGXSOCKETCORES=24
DGXNSOCKET=2

if [[ $SLURM_NNODES -gt 1 ]]; then
  export MULTI_NODE="--nnodes=$SLURM_NNODES --node_rank=\$SLURM_NODEID --master_addr=$MASTER_IP --master_port=$PORT"
else
  export MULTI_NODE=""
fi

# Diagnostic prints
echo "Hosts: $hosts"
echo "Master IP: $MASTER_IP"
echo "Port: $PORT"
echo "Slurm num nodes: $SLURM_NNODES"


LAUNCH_CMD_P1="python -m torch.distributed.launch \
    --nproc_per_node=16 ${MULTI_NODE} run_pretrain.py \
    --model_name=PYT_lamb_electra_base_pretrain_seq_128_65536_lr6e-3_wup_2000_b1_0.878_b2_0.974_decay_0.5_total10000_0817 \
    --data_dir=pretraining_data/ \
    --pretrain_hdf5=pretraining_data/pretrain_hdf5_seq_128/*.hdf5 \
    --num_train_steps=10000 \
    --num_warmup_steps=2000 \
    --disc_weight=50.0 \
    --generator_hidden_size=0.3333333 \
    --learning_rate=6e-3 \
    --train_batch_size=128 \
    --max_seq_length=128 \
    --save_checkpoints_steps=1000 \
    --keep_checkpoint_max=5 \
    --log_freq=2 \
    --optimizer=lamb \
    --skip_adaptive \
    --opt_beta_1=0.878 \
    --opt_beta_2=0.974 \
    --lr_decay_power=0.5 \
    --gradient_accumulation_steps=4 \
    --restore_checkpoint=latest "

LAUNCH_CMD_P2="python -m torch.distributed.launch \
    --nproc_per_node=16 ${MULTI_NODE} run_pretrain.py \
    --model_name=PYT_lamb_electra_base_pretrain_seq_512_30720_lr_4e-3_wup_200_decay_0.5_skip_adpative_yes_end_lr_0.0_total1670_0817 \
    --data_dir=pretraining_data/ \
    --pretrain_hdf5=pretraining_data/pretrain_hdf5_seq_512/*.hdf5 \
    --num_train_steps=1670 \
    --num_warmup_steps=200 \
    --disc_weight=50.0 \
    --generator_hidden_size=0.3333333 \
    --learning_rate=4e-3 \
    --train_batch_size=24 \
    --max_seq_length=512 \
    --save_checkpoints_steps=200 \
    --keep_checkpoint_max=5 \
    --log_freq=10 \
    --optimizer=lamb \
    --skip_adaptive \
    --lr_decay_power=0.5 \
    --end_lr=0.0 \
    --gradient_accumulation_steps=10 \
    --phase2 \
    --restore_checkpoint=pretraining_data/models/PYT_lamb_electra_base_pretrain_seq_128_65536_lr6e-3_wup_2000_b1_0.878_b2_0.974_decay_0.5_total10000_0817/checkpoints/ckpt-10000  "



srun --mpi=pmi2 -l --container-image="${docker_image}" --container-mounts="${mounts}" bash -c "${LAUNCH_CMD_P2}"