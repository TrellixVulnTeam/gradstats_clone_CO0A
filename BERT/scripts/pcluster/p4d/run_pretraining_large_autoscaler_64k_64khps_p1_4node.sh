#!/bin/bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

# setup NCCL to use EFA
# Trial script for 1 node
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
export RDMAV_FORK_SAFE=1
export NCCL_TREE_THRESHOLD=0
# export OMP_NUM_THREADS=96

# setup conda
source /fsx/conda/etc/profile.d/conda.sh
conda activate /fsx/conda/envs/pytorch_latest_p37_fsx

# 64K batch settings for 32 40GB GPUs (4 P4D)
train_batch_size=${1:-2048}
learning_rate=${2:-"1.37E-03"}
adamw_beta1=0.95238
adamw_beta2=0.86471
adamw_weight_decay=0.19891
adamw_eps="1.0e-11"
lr_poly_power=1
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2222"}
train_steps=${6:-14063}
save_checkpoint_steps=${7:-5}
# for elastic setup this should be true by default
resume_training=${8:-"true"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-32}
seed=${12:-72337}
job_name=${13:-"bert_large_adamw_pretraining_autoscaler"}
allreduce_post_accumulation=${14:-"true"}
# NOTE: this phase2 bs is different from NV training setup where phase2 bs is half of phase1
train_batch_size_phase2=${16:-1024}
learning_rate_phase2=${17:-"2.8464e-4"}
adamw_phase2_beta1=0.65322
adamw_phase2_beta2=0.82451
adamw_phase2_weight_decay=0.19891
warmup_proportion_phase2=${18:-"0.5"}
train_steps_phase2=${19:-1562}
gradient_accumulation_steps_phase2=${20:-128}
sampling_with_replacement=${21:-"false"}
enable_autoscaler=${22:-"true"}
AUTOSCALER_CONFIG=/fsx/code/gradstats/BERT/autoscaler.yaml
DATASET=books_wiki_en_corpus
DATA_DIR_PHASE1=/fsx/data/nlp/BERT/phase1/
BERT_CONFIG=/fsx/code/gradstats/BERT/bert_config.json
DATASET2=books_wiki_en_corpus
DATA_DIR_PHASE2=/fsx/data/nlp/BERT/phase2/
CODEDIR=${23:-"/fsx/mzanur/gradstats/BERT"}
init_checkpoint=${24:-"None"}
RESULTS_DIR=/fsx/logs/BERT/64kbs_64hps_autoscaler/
CHECKPOINTS_DIR=$RESULTS_DIR/checkpoints

mkdir -p $CHECKPOINTS_DIR


if [ ! -d "$DATA_DIR_PHASE1" ] ; then
   echo "Warning! $DATA_DIR_PHASE1 directory missing. Training cannot start"
fi
if [ ! -d "$RESULTS_DIR" ] ; then
   echo "Error! $RESULTS_DIR directory missing."
   exit -1
fi
if [ ! -d "$CHECKPOINTS_DIR" ] ; then
   echo "Warning! $CHECKPOINTS_DIR directory missing."
   echo "Checkpoints will be written to $RESULTS_DIR instead."
   CHECKPOINTS_DIR=$RESULTS_DIR
fi
if [ ! -f "$BERT_CONFIG" ] ; then
   echo "Error! BERT configuration file not found at $BERT_CONFIG"
   exit -1
fi
#
PREC=""
if [ "$precision" = "fp16" ] ; then
   PREC="--fp16"
elif [ "$precision" = "fp32" ] ; then
   PREC=""
elif [ "$precision" = "tf32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

ACCUMULATE_GRADIENTS=""
if [ "$accumulate_gradients" == "true" ] ; then
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps"
fi

CHECKPOINT=""
if [ "$resume_training" == "true" ] ; then
   CHECKPOINT="--resume_from_checkpoint"
fi

SAMPLING_WITH_REPLACEMENT=""
if [ "$sampling_with_replacement" == "true" ] ; then
   SAMPLING_WITH_REPLACEMENT="--sampling_with_replacement"
fi

ENABLE_AUTOSCALER=""
if [ "$enable_autoscaler" == "true" ] ; then
   ENABLE_AUTOSCALER="--enable-autoscaler"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

INIT_CHECKPOINT=""
if [ "$init_checkpoint" != "None" ] ; then
   INIT_CHECKPOINT="--init_checkpoint=$init_checkpoint"
fi

echo $DATA_DIR_PHASE1
INPUT_DIR=$DATA_DIR_PHASE1
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE1"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size"
CMD+=" --max_seq_length=128"
CMD+=" --max_predictions_per_seq=20"
CMD+=" --max_steps=$train_steps"
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --use_adamw"
CMD+=" --learning_rate=$learning_rate"
CMD+=" --adamw_beta1=$adamw_beta1"
CMD+=" --adamw_beta2=$adamw_beta2"
CMD+=" --adamw_weight_decay=$adamw_weight_decay"
CMD+=" --adamw_eps=$adamw_eps"
CMD+=" --lr_poly_power=$lr_poly_power"
CMD+=" --seed=$seed"
CMD+=" --autoscaler-cfg-path=$AUTOSCALER_CONFIG"
CMD+=" --disable_progress_bar"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" $SAMPLING_WITH_REPLACEMENT"
CMD+=" $ENABLE_AUTOSCALER"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --label bert_training_large_64K_4node "

# set up environment variables for Torch DistributedDataParallel - set by PyTorchJob
PROC_PER_NODE=8
WORLD_SIZE=$SLURM_NTASKS
RANK=$SLURM_NODEID
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12277"

echo "WORLD SIZE $WORLD_SIZE"
echo "RANK $RANK"
echo "MASTER_ADDR_JOB $MASTER_ADDR_JOB"

# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO

# Note: If we have 4 nodes in cluster, we will launch 1 Master and 3 Workers in EKS launcher - WORLD_SIZE will be set as 4 and we will pass 8 gpus per node
# CMD="python -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} $CMD"
CMD="/fsx/conda/envs/pytorch_latest_p37_fsx/bin/python -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR_JOB} --master_port=${MASTER_PORT_JOB} $CMD"


if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase1_%s_gbs%d" "$precision" $GBS
  DATESTAMP=`date +'%y%m%d%H%M%S'`
  LOGFILE=$RESULTS_DIR/$job_name.$TAG.$DATESTAMP.log
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

echo "finished phase 1 pretraining"
