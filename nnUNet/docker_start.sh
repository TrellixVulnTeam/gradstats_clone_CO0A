# CONTAINER_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
CONTAINER_IMAGE=427566855058.dkr.ecr.us-east-1.amazonaws.com/unet:aws-unet
CONTAINER_NAME=aws_unet_test
docker run --rm -it -d --gpus all \
                    --name $CONTAINER_NAME \
                    --net=host --uts=host --ipc=host \
                    --ulimit stack=67108864 --ulimit memlock=-1 \
                    --security-opt seccomp=unconfined \
		    -v /home/ubuntu/gradstats:/mnt \
                    -v /home/ubuntu/data:/data \
                    $CONTAINER_IMAGE