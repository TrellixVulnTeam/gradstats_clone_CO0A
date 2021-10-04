
ACCOUNT=763104351884
REGION=us-east-1
TAG=1.8.1-gpu-py36-cu111-ubuntu18.04
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com


CONTAINER_NAME=bert
CODE_MOUNT="-v /fsx/:/fsx"

IMAGE="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:${TAG}"
#763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:


docker run --runtime=nvidia --gpus 8  \
   	--privileged \
   	--rm -d   \
  	--net=host --uts=host --ipc=host \
  	--ulimit stack=67108864 --ulimit memlock=-1  --security-opt seccomp=unconfined \
   	--name $CONTAINER_NAME \
	${CODE_MOUNT} \
	${IMAGE} \
   	bash -c  '"sleep infinity"'


docker exec -it ${CONTAINER_NAME} bash -c "pip install nltk html2text progressbar onnxruntime git+https://github.com/NVIDIA/dllogger" 
docker exec -it ${CONTAINER_NAME} bash -c  "pip install tqdm tensorboard yacs"
