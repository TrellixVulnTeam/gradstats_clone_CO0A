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
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
export RDMAV_FORK_SAFE=1
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=ens5
export OMP_NUM_THREADS=96

# 64K batch settings for 8 32GB GPUs
train_batch_size=${1:-8192}
learning_rate=${2:-"5.9415e-4"}
adamw_beta1=0.934271
adamw_beta2=0.989295
adamw_weight_decay=0.31466
adamw_eps="1.0e-11"
lr_poly_power=1
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2222"}
train_steps=${6:-10}
save_checkpoint_steps=${7:-5}
# for elastic setup this should be true by default
resume_training=${8:-"true"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-128}
seed=${12:-72337}
job_name=${13:-"bert_large_adamw_pretraining"}
allreduce_post_accumulation=${14:-"true"}
# NOTE: this phase2 bs is different from NV training setup where phase2 bs is half of phase1
train_batch_size_phase2=${16:-1024}
learning_rate_phase2=${17:-"2.8464e-4"}
adamw_phase2_beta1=0.963567
adamw_phase2_beta2=0.952647
adamw_phase2_weight_decay=0.31466
warmup_proportion_phase2=${18:-"0.5"}
train_steps_phase2=${19:-1562}
gradient_accumulation_steps_phase2=${20:-128}
sampling_with_replacement=${21:-"true"}
enable_autoscaler=${22:-"true"}
AUTOSCALER_CONFIG=/home/ubuntu/workspace/gradstats/BERT/autoscaler.yaml
DATASET=books_wiki_en_corpus
DATA_DIR_PHASE1=/home/ubuntu/data/nlp/BERT/phase1/ 
BERT_CONFIG=/home/ubuntu/workspace/gradstats/BERT/bert_config.json
DATASET2=books_wiki_en_corpus 
DATA_DIR_PHASE2=/home/ubuntu/data/nlp/BERT/phase2/ 
CODEDIR=${23:-"/home/ubuntu/workspace/gradstats/BERT"}
init_checkpoint=${24:-"None"}
RESULTS_DIR=/home/ubuntu/logs/BERT/2x_debug/
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
 
#Start Phase2
echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-large-uncased"
CMD+=" --train_batch_size=$train_batch_size_phase2"
CMD+=" --max_seq_length=512"
CMD+=" --max_predictions_per_seq=80"
CMD+=" --max_steps=$train_steps_phase2"
CMD+=" --warmup_proportion=$warmup_proportion_phase2"
CMD+=" --num_steps_per_checkpoint=$save_checkpoint_steps"
CMD+=" --learning_rate=$learning_rate_phase2"
CMD+=" --use_adamw"
CMD+=" --adamw_beta1=$adamw_phase2_beta1"
CMD+=" --adamw_beta2=$adamw_phase2_beta2"
CMD+=" --adamw_weight_decay=$adamw_weight_decay"
CMD+=" --adamw_eps=$adamw_eps"
CMD+=" --lr_poly_power=$lr_poly_power"
CMD+=" --seed=$seed"
CMD+=" --disable_progress_bar"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $ENABLE_AUTOSCALER"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $SAMPLING_WITH_REPLACEMENT"
#CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
# resume from latest ckpt
CMD+=" --do_train --phase2 --resume_from_checkpoint " 
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --label bert_training_large_64k_local "

# CMD="python -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} $CMD"
CMD="python -m torch.distributed.launch --nproc_per_node=8 $CMD"

if [ "$create_logfile" = "true" ] ; then
  export GBS=$(expr $train_batch_size_phase2 \* $num_gpus)
  printf -v TAG "pyt_bert_pretraining_phase2_%s_gbs%d" "$precision" $GBS
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

echo "finished pretraining phase2"

