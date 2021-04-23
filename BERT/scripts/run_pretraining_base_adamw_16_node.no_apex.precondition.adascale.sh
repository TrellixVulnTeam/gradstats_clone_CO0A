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

# echo "Container nvidia build = " $NVIDIA_BUILD_ID
# conda env 
source /shared/conda/bin/activate /shared/conda/envs/adascale_bert/
# 128K GBS settings
train_batch_size=${1:-1024}
learning_rate=${2:-"1.3653e-3"}
adamw_beta1=0.952378
adamw_beta2=0.86471
adamw_weight_decay=0.19891
adamw_eps="1.0e-11"
lr_poly_power=2
precision=${3:-"fp16"}
num_gpus=${4:-8}
warmup_proportion=${5:-"0.2843"}
#train_steps=${6:-7038}
train_steps=${6:-200}
save_checkpoint_steps=${7:-100}
resume_training=${8:-"false"}
create_logfile=${9:-"true"}
accumulate_gradients=${10:-"true"}
gradient_accumulation_steps=${11:-64}
seed=${12:-12439}
job_name=${13:-"bert_adamw_pretraining"}
#allreduce_post_accumulation=${14:-"true"}
allreduce_post_accumulation=${14:-"false"}
#allreduce_post_accumulation_fp16=${15:-"true"}
allreduce_post_accumulation_fp16=${15:-"false"}
train_batch_size_phase2=${16:-512}
learning_rate_phase2=${17:-"6.1951e-5"}
adamw_phase2_beta1=0.65322
adamw_phase2_beta2=0.82451
adamw_phase2_weight_decay=0.19891
#warmup_proportion_phase2=${18:-"0.5"}
warmup_proportion_phase2=${18:-"0.25"}
#train_steps_phase2=${19:-781}
train_steps_phase2=${19:-1563}
gradient_accumulation_steps_phase2=${20:-128}
DATASET=books_wiki_en_corpus #hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
DATA_DIR_PHASE1=/shared/data/nlp/BERT/phase1/ #${21:-$BERT_PREP_WORKING_DIR/${DATASET}/}
BERT_CONFIG=/shared/scaling_without_tuning/BERT_adascale/bert_base_config.json
DATASET2=books_wiki_en_corpus # hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en # change this for other datasets
DATA_DIR_PHASE2=/shared/data/nlp/BERT/phase2/ #${22:-$BERT_PREP_WORKING_DIR/${DATASET2}/}
CODEDIR=${23:-"/shared/scaling_without_tuning/BERT_adascale/"}
init_checkpoint=${24:-"None"}
RESULTS_DIR=$CODEDIR/results/pretrain_base_adamw_16node_gns_autoamp_precond
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
   echo "Error! BERT base configuration file not found at $BERT_CONFIG"
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
CMD+=" --bert_model=bert-base-uncased"
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
CMD+=" --disable_progress_bar"
CMD+=" --enable_gns"
CMD+=" --use_adascale"
CMD+=" --lr_scale=2.0"
CMD+=" --gns_smoothing=0.25"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" $INIT_CHECKPOINT"
CMD+=" --do_train"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --use_preconditioner "
# # set up environment variables for Torch DistributedDataParallel
WORLD_SIZE=$SLURM_NTASKS
RANK=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12244"
 
# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
 
 
CMD="/shared/conda/envs/adascale_bert/bin/python3 -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR_JOB} --master_port=${MASTER_PORT_JOB} $CMD"


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

echo "finished pretraining"

#Start Phase2

precision="fp16"

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
   ACCUMULATE_GRADIENTS="--gradient_accumulation_steps=$gradient_accumulation_steps_phase2"
fi

ALL_REDUCE_POST_ACCUMULATION=""
if [ "$allreduce_post_accumulation" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION="--allreduce_post_accumulation"
fi

ALL_REDUCE_POST_ACCUMULATION_FP16=""
if [ "$allreduce_post_accumulation_fp16" == "true" ] ; then
   ALL_REDUCE_POST_ACCUMULATION_FP16="--allreduce_post_accumulation_fp16"
fi

echo $DATA_DIR_PHASE2
INPUT_DIR=$DATA_DIR_PHASE2
CMD=" $CODEDIR/run_pretraining.py"
CMD+=" --input_dir=$DATA_DIR_PHASE2"
CMD+=" --output_dir=$CHECKPOINTS_DIR"
CMD+=" --config_file=$BERT_CONFIG"
CMD+=" --bert_model=bert-base-uncased"
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
CMD+=" --enable_gns"
CMD+=" --use_adascale"
CMD+=" --lr_scale=2.0"
CMD+=" --gns_smoothing=0.25"
CMD+=" $PREC"
CMD+=" $ACCUMULATE_GRADIENTS"
CMD+=" $CHECKPOINT"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION"
CMD+=" $ALL_REDUCE_POST_ACCUMULATION_FP16"
CMD+=" --do_train --phase2 --resume_from_checkpoint --phase1_end_step=$train_steps"
CMD+=" --json-summary ${RESULTS_DIR}/dllogger.json "
CMD+=" --use_preconditioner "

# CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"

CMD="/shared/conda/envs/adascale_bert/bin/python3 -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=${RANK} --master_addr=${MASTER_ADDR_JOB} --master_port=${MASTER_PORT_JOB} $CMD"

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

echo "finished phase2"
