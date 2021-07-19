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
    # parser.add_argument("--per_gpu_train_batch_size", type=int, default=3)

    args = parser.parse_args()
    os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NCCL_SOCKET_IFNAME'] = 'ens5'
    # environment prameter parsed from sagemaker
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    print(f'current rank is {rank}')
    if args.num_nodes >= 2:
        cmd = f"python -m torch.distributed.launch " \
              f"--nnodes={args.num_nodes} " \
              f"--node_rank={rank} " \
              f"--nproc_per_node={num_gpus} " \
              f"--master_addr={hosts[0]} " \
              f"--master_port={args.port} " \
              f"run_qa.py " \
              f"--model_name_or_path albert-base-v2 " \
              f"--dataset_name squad " \
              f"--cache_dir . "\
              f"--do_train " \
              f"--do_eval " \
              f"--learning_rate 3e-5 " \
              f"--num_train_epochs 2 " \
              f"--max_seq_length 384 " \
              f"--doc_stride 128 " \
              f"--output_dir {args.output_dir} " \
              f"--per_device_eval_batch_size=3 " \
              f"--per_device_train_batch_size=3 "

    else:
        cmd = f"python -m torch.distributed.launch --nproc_per_node={num_gpus} " \
              f"scripts/train.py --gpus 8 --fold 0 --dim 2 --amp | tee {args.output_dir}/train_nvidia_1x_bs6x_lrscale6x_adam.log"

    try:
        sb.run(cmd, shell=True)
    except Exception as e:
        print(e)
    
    print('distributed script ending...')
