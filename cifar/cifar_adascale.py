# Source: https://leimao.github.io/blog/PyTorch-Distributed-Training/
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np
from fairscale.optim import AdaScale
import time
import statistics
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def evaluate(model, device, test_loader):

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main():
    num_epochs_default = 1
    batch_size_default = 256  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"
    adascale_scale_default = 1
    weight_decaye_default = 1e-5
    momentum_default = 0.9
    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--weight_decay", type=float, help="Weight decay.", default=weight_decaye_default)
    parser.add_argument("--momentum", type=float, help="Momentum.", default=momentum_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    parser.add_argument("--use_adascale", action="store_true", help="Use adascale optimizer for training.")
    parser.add_argument("--adascale_scale", type=int, help="Scale factor for adascale.", default=adascale_scale_default)

    parser.add_argument("--use_fp16_compress", action="store_true", help="Use fp16 compression for training.")
    parser.add_argument('--log_dir',
                        default='./logs',
                        type=str,
                        help='log directory path.')

    argv = parser.parse_args()

    local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume
    use_adascale = argv.use_adascale
    use_fp16_compress = argv.use_fp16_compress
    tensorboard_path = f'{argv.log_dir}/worker-0'
    weight_decay = argv.weight_decay
    momentum = argv.momentum
    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''
    print(" Adascale hyperparameters")
    print(" Base batch size: ", batch_size)
    print(" Scale factor batch size: ", get_world_size())
    print(" Batch size after scaling: ", batch_size*get_world_size())
    print(" Number of steps : ")

    if get_rank() == 0:
        # tensorboard summary writer (by default created for all workers)
        writer = SummaryWriter(tensorboard_path)


    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend="nccl")
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=False)

    device = torch.device("cuda:{}".format(local_rank))
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    if use_fp16_compress:
        ddp_model.register_comm_hook(state=None, hook=fp16_compress_hook)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    # Restricts data loading to a subset of the dataset exclusive to the current process
    train_sampler = DistributedSampler(dataset=train_set)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler, num_workers=8)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=8)

    criterion = nn.CrossEntropyLoss()
    if use_adascale:
        print(" INFO: Using Adascale ")
        optimizer = AdaScale(optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay))
    else:
        print(" INFO: Not using Adascale")
        optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    step_times = []
    # # Loop over the dataset multiple times
    step = 0
    done = False
    epoch = 0
    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        # Save and evaluate model routinely
        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
                if get_rank() == 0:
                    writer.add_scalar(f'Val/Acc', accuracy, epoch)
                    writer.flush()
                torch.save(ddp_model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        ddp_model.train()

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            if get_rank() == 0:
                writer.add_scalar(f'Train/Loss', accuracy, step)
                writer.flush()
            optimizer.step()
            step += 1


            end = time.time()
    print(" INFO: Total steps: ", step)



if __name__ == "__main__":
    main()
