CONTAINER_NAME=bert
CODE_MOUNT="-v /fsx/:/fsx"

IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04


docker run --runtime=nvidia --gpus 8  \
   --privileged \
   --rm -d   \
   --name $CONTAINER_NAME \
	${CODE_MOUNT} \
	    ${IMAGE} \
   bash -c  '"sleep infinity"'


docker exec -it ${CONTAINER_NAME} bash -c "pip install nltk html2text progressbar onnxruntime git+https://github.com/NVIDIA/dllogger" 
docker exec -it ${CONTAINER_NAME} bash -c  "pip install tqdm tensorboard yacs"
