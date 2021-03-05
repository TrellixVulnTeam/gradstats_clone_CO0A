#!/bin/bash

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE=$SLURM_NTASKS
RANK=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"

# training details
BASE_DIR=/shared/PyTorchBenchmarks/benchmarks/mmclassification_adascale
TRAIN_CONFIG=$BASE_DIR/configs/imagenet/resnext50_32x4d_b32x16_adascale.py

# logs
LOG_DIR=/shared/training_logs

# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
#export NCCL_TREE_THRESHOLD=0
#export NCCL_SOCKET_IFNAME=eth0

# export LD_LIBRARY_PATH=/home/ubuntu/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH

# conda env - is this required? already part of bashrc
source /shared/conda/bin/activate

# main training launch command that uses DDP - set scale correctly!!
/shared/conda/bin/python -m torch.distributed.launch \
	--nproc_per_node=$PROC_PER_NODE \
	--nnodes=$WORLD_SIZE \
	--node_rank=$RANK \
	--master_addr=${MASTER_ADDR_JOB} \
	--master_port=${MASTER_PORT_JOB} \
	$BASE_DIR/tools/train.py $TRAIN_CONFIG --scale 32.0 --launcher="pytorch" | tee $LOG_DIR/train.$RANK.$WORLD_SIZE.log

