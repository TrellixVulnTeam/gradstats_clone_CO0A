#!/bin/bash

# setup NCCL to use EFA
# export FI_PROVIDER=efa
# export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
export RDMAV_FORK_SAFE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=48

train_batch_size=${1:-256}
learning_rate=${2:-"0.001"}
enable_amp=${3:-"--amp"}
num_gpus=${4:-4}
resume_training=${5:-"false"}
NHWC=${6:-"--channels-last"}
ARCH=${7:-"resnet50"}
NUM_WORKERS=${8:-12}
TOTAL_EPOCHS=${9:-90}
create_logfile="true"
DATA_DIR="/shared/benchmarking_datasets/imagenet/processed"
CODEDIR="/gradstats/resnet50/imagenet"
RESULTS_DIR="/shared/export/resnet_adamw_g4_delme"
if [ ! -d "$DATA_DIR" ] ; then
   echo "Warning! $DATA_DIR directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Results directory does not exist. Creating $RESULTS_DIR"
   mkdir -p $RESULTS_DIR
fi

CMD=" $CODEDIR/trainer_ddp_amp.py"
CMD+=" -a $ARCH"
CMD+=" $DATA_DIR"
CMD+=" --batch-size $train_batch_size"
CMD+=" --workers $NUM_WORKERS"
CMD+=" $enable_amp"
CMD+=" --optimizer AdamW"
CMD+=" --lr $learning_rate"
CMD+=" --weight-decay 0.1"
CMD+=" --autoscaler_cfg /gradstats/resnet50/imagenet/autoscaler.yaml" 
CMD+=" --epochs $TOTAL_EPOCHS"
CMD+=" $NHWC"

# Note: If we have 4 nodes in cluster, we will launch 1 Master and 3 Workers in EKS launcher - WORLD_SIZE will be set as 4 and we will pass 8 gpus per node 
# For EKS we set 8 GPUs per node (pod)
PROC_PER_NODE=4

CMD="python -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} $CMD"

if [ "$create_logfile" = "true" ] ; then
  printf -v TAG "pyt_resnet50_ddp_amp_gbs%d" $train_batch_size
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$TAG.$DATESTAMP.log
  printf "Logs written to %s\n" "$LOGFILE"
fi

set -x
if [ -z "$LOGFILE" ] ; then
   $CMD
else
   (
     $CMD
   ) |& tee $LOGFILE
fi

set +x

echo "finished training"
