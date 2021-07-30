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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'radam', 'sgd'], default='adam')
    parser.add_argument("--label", type=str, default="unet_train", help="name of results folder and sagemaker job")
    # adascale related
    parser.add_argument('--enable_adascale', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--lr_scale',
        type=float,
        default=1.0,
        help='Batch scaling factor for AdaScale.')
    parser.add_argument('--enable_gns', type=str, choices=['True', 'False'], default='False')
    parser.add_argument('--gns_smoothing',
        type=float,
        default=0.0,
        help='Smoothing factor for gradient stats.')
    
    args = parser.parse_args()
    os.environ['NCCL_DEBUG'] = 'INFO'
    # environment prameter parsed from sagemaker
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    num_workers = int(os.environ["SM_NUM_CPUS"]) // num_gpus
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
              f"--gpus {num_gpus} " \
              f"--data {train_data_path} " \
              f"--results . " \
              f"--num_nodes {args.num_nodes} " \
              f"--amp " \
              f"--num_workers {num_workers} " \
              f"--batch_size {args.batch_size} " \
              f"--val_batch_size 64 " \
              f"--learning_rate {args.learning_rate} " \
              f"--optimizer {args.optimizer} " \
              f"--dim 2 " \
              f"--label  {args.label} "
        if args.enable_adascale == "True":
            cmd += f" --enable_adascale --lr_scale {args.lr_scale} "
        if args.enable_gns == "True":
            cmd += f" --enable_gns --gns_smoothing {args.gns_smoothing} "

        try:
            sb.Popen(cmd, shell=True)
        except Exception as e:
            print(e)
    
    print('distributed script ending...')
