docker exec unet_nvidia_test /bin/bash -c "python /mnt/nnUNet/scripts/train.py --gpus 8 --fold 0 --dim 2 --amp" | tee train_nvidia_1x_bs6x_lrscale6x_adam.log
