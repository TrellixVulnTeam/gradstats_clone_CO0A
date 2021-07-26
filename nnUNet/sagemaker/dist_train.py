import argparse
import os
import json
import subprocess as sb
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get model info')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--master_addr', type=str, help='master ip address')
    parser.add_argument('--port', type=str, default='1234', help='master port to use')
    parser.add_argument('--model_type', type=str, default='unet_2d')
    parser.add_argument('--platform', type=str, default='SM')

    args = parser.parse_args()
    os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    # environment prameter parsed from sagemaker
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    print(f'current node rank is {rank}')
    os.environ['MASTER_ADDR'] = hosts[0]
    os.environ['MASTER_PORT'] = args.port
    os.environ['WORLD_SIZE'] = str(args.num_nodes)
    os.environ['NODE_RANK'] = str(rank)

    train_data_path = os.environ['SM_CHANNEL_TRAIN']

    for i in range(num_gpus):
        os.environ['LOCAL_RANK'] = str(i)
        cmd = f"python main_tfboard.py " \
              f"--exec_mode train " \
              f"--task 01 " \
              f"--fold 0 " \
              f"--gpus 8 " \
              f"--data {train_data_path} " \
              f"--results . " \
              f"--num_nodes {args.num_nodes} " \
              f"--amp " \
              f"--num_workers 12 " \
              f"--batch_size 64 " \
              f"--val_batch_size 64 " \
              f"--learning_rate 0.001 " \
              f"--optimizer adam " \
              f"--enable_adascale " \
              f"--lr_scale 6.0 " \
              f"--dim 2 " \
              f"--label unet_6x_adam_adascale_fold0_adaptive-off_run1 " \
              f"| tee {args.output_dir}unet_6x_adam_adascale_fold0_adaptive-off_run1.log"

        try:
            sb.Popen(cmd, shell=True)
        except Exception as e:
            print(e)
    
    print('distributed script ending...')
