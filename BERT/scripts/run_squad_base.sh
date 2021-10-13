#!/usr/bin/env bash

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
source /fsx/conda/bin/activate /home/ubuntu/anaconda3/envs/pytorch_latest_p37/

code_dir="/fsx/code/gradstats/BERT"
#init_checkpoint=${1:-"$code_dir/results/pretrain_base_4/checkpoints/ckpt_8601.pt"}
init_checkpoint=${1:-"$code_dir/results/pretrain_base_2node_adam/checkpoints/ckpt_8601.pt"} 
# init_checkpoint="/shared/scaling_without_tuning/BERT/results/pretrain_base_adamw_4node_gns_autoamp_precond/checkpoints/ckpt_8601.pt"
# checkpoints/ckpt_7038.pt"}
RESULTS_DIR=${2:-"/fsx/code/gradstats/BERT/results/pretrain_base_2node_adam"}
epochs=${3:-"2.0"}
batch_size=${4:-"4"}
learning_rate=${5:-"3e-5"}
precision=${6:-"fp16"}
num_gpu=${7:-"8"}
seed=${8:-"1"}
squad_dir=${9:-"/fsx/data/nlp/SQUAD/download/squad/v1.1"}
vocab_file=${10:-"/fsx/data/nlp/SQUAD/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt"}
mode=${11:-"train eval"}
CONFIG_FILE=${12:-"/fsx/code/gradstats/BERT/bert_base_config.json"}
max_steps=${13:-"-1"}
OUT_DIR=${14:-"$RESULTS_DIR/finetune"}

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [ "$num_gpu" = "1" ] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="/home/ubuntu/anaconda3/envs/pytorch_latest_p37/bin/python3  $mpi_command /fsx/code/gradstats/BERT/run_squad.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
  CMD+="--lr_scale=1.0 "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-base-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" $use_fp16"
CMD+=" --json-summary=$OUT_DIR/dllogger.json "
LOGFILE=$OUT_DIR/logfile.txt
echo "$CMD |& tee $LOGFILE"
time $CMD |& tee $LOGFILE
