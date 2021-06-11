docker exec unet_nvidia_test /bin/bash -c "python /mnt/scripts/train.py --gpus 8 --fold 0 --dim 2 --amp" | tee train_nvidia_6x.log
