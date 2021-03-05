#!/bin/bash

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE=$SLURM_NTASKS
RANK=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12244"

# training details
BASE_DIR=/shared/scaling_without_tuning/maskrcnn
TRAIN_CONFIG=$BASE_DIR/configs/e2e_mask_rcnn_R_50_FPN_1x_giou_novo_ls.yaml
PATH_CONFIG=$BASE_DIR/maskrcnn_benchmark/config/paths_catalog.py

# base schedule (note that DDP base schedule warmup is maintained)
BASE_LR=0.008
MAX_ITER=45000
WARMUP_FACTOR=0.0001
WARMUP_ITERS=100
WEIGHT_DECAY=5e-4
OPTIMIZER="UnfusedNovoGrad"
BETA1=0.9
BETA2=0.35
LS=0.1
# ENABLE CLIP GRAD VALUE FOR NovoGrad 512 with AdaScale
# GRADIENT_CLIP_NORM=1.5
GRADIENT_CLIP_NORM=0.0

# update train and test ims per batch based on scale 
TRAIN_IMS_PER_BATCH=64
TEST_IMS_PER_BATCH=16
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=24
NPROC_PER_NODE=8
LR_SCHEDULE="COSINE"
# update scale
LR_SCALE=2.0

# logs
LOG_DIR=/shared/training_logs

# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO

# conda env 
source /shared/conda/bin/activate /shared/conda/envs/adascale_fp16/


# main training launch command that uses DDP
/shared/conda/envs/adascale_fp16/bin/python -m torch.distributed.launch \
 --nproc_per_node=$PROC_PER_NODE \
 --nnodes=$WORLD_SIZE \
 --node_rank=$RANK \
 --master_addr=${MASTER_ADDR_JOB} \
 --master_port=${MASTER_PORT_JOB} \
${BASE_DIR}/tools/train_mlperf.py --config-file ${TRAIN_CONFIG} \
 PATHS_CATALOG ${PATH_CONFIG} \
 DISABLE_REDUCED_LOGGING True \
 SOLVER.BASE_LR ${BASE_LR} \
 SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
 SOLVER.MAX_ITER ${MAX_ITER} \
 SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
 SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
 SOLVER.WEIGHT_DECAY_BIAS 0 \
 SOLVER.WARMUP_METHOD linear \
 SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
 SOLVER.OPTIMIZER ${OPTIMIZER} \
 SOLVER.BETA1 ${BETA1} \
 SOLVER.BETA2 ${BETA2} \
 SOLVER.GRADIENT_CLIP_NORM ${GRADIENT_CLIP_NORM} \
 MODEL.RPN.LS ${LS} \
 SOLVER.LR_SCHEDULE ${LR_SCHEDULE} \
 SOLVER.LR_SCALE ${LR_SCALE} \
 TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
 NHWC True \
 DTYPE 'float16' | tee $LOG_DIR/train_mrcnn.novo.$RANK.$WORLD_SIZE.log
