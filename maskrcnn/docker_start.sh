CONTAINER_IMAGE=427566855058.dkr.ecr.us-east-1.amazonaws.com/gradstats_mrcnn:latest
CONTAINER_NAME=maskrcnn_test
docker run --rm -it -d --gpus all \
                    --name $CONTAINER_NAME \
                    --net=host --uts=host --ipc=host \
                    --ulimit stack=67108864 --ulimit memlock=-1 \
                    --security-opt seccomp=unconfined \
		    -v $(pwd):/mnt \
                    -v /home/ubuntu/data:/data \
                    $CONTAINER_IMAGE
