CONTAINER_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
CONTAINER_NAME=resnet_test
docker run --rm -it -d --gpus all \
                    --name $CONTAINER_NAME \
                    --net=host --uts=host --ipc=host \
                    --ulimit stack=67108864 --ulimit memlock=-1 \
                    --security-opt seccomp=unconfined \
		    -v $(pwd):/mnt \
                    -v /home/ubuntu/autoscaler_data/data:/data \
                    $CONTAINER_IMAGE
