import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from with_replacement_sampler import ReplacementDistributedSampler
import numpy as np
import math
from automl.autoscaler import AdaScale
from automl.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from utils import upload_dir, make_path_if_not_exists, read_s3_textfile
from datetime import timedelta


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# global step needs to be part of checkpoint - to keep track of progress when
# cluster resize occurs

global_step = 0 

#TODO: Refactor this class - generalize for all models
class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, arch, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.arch = arch
        self.model = model
        self.optimizer = optimizer
        self.global_step = 0

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "arch": self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """
        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])
        self.global_step = obj["global_step"]

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)


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


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    return rt


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('-a',
                        '--arch',
                        metavar='ARCH',
                        default='resnet18',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet18)')

    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Enable mixed precision training")

    parser.add_argument('--use_preconditioner',
                        default=False,
                        action='store_true',
                        help='condition gradients with moving average stats')

#    parser.add_argument('--autoscaler-cfg-path-template',
#                        default="/gradstats/resnet50/imagenet/autoscaler_adam_{}x.yaml",
#                        type=str,
#                        help='AutoScaler configuration template')

    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs',
                        default=90,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        metavar='N',
                        help='batch size per worker')

    parser.add_argument('--gradient-accumulation-steps',
                        default=1,
                        type=int,
                        dest='gradient_accumulation_steps',
                        help='set to > 1 for larger batch sizes')

    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')

    parser.add_argument('--optimizer',
                        default="SGD",
                        type=str,
                        metavar='O',
                        help='Optimizer, one of SGD or AdamW')

    parser.add_argument('--beta1',
                        default=0.9,
                        type=float,
                        help='adamw beta1 (default: 0.9)',
                        dest='beta1')

    parser.add_argument('--beta2',
                        default=0.999,
                        type=float,
                        help='adamw beta2 (default: 0.999)',
                        dest='beta2')

    parser.add_argument('--eps',
                        default=1e-8,
                        type=float,
                        help='adamw eps (default: 1e-8)',
                        dest='eps')

    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        metavar='M',
                        help='momentum')

    parser.add_argument('--wd',
                        '--weight-decay',
                        default=1e-4,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-p',
                        '--print-freq',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')

    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--run-gns-experiment',
                        action='store_true',
                        help='when enabled we replace step decay with linear decay to enable different batch size runs')

    parser.add_argument('--enable-autoscaler',
                        action='store_true',
                        help='when enabled we start measuring gradient stats')

    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')

    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('--rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')

    parser.add_argument('--dist-url',
                        default='env://',
                        type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--dist-backend',
                        default='nccl',
                        type=str,
                        help='distributed backend')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')

    # DO NOT PASS THIS EXPLICITLY FOR ELASTIC CASE
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--channels-last",
                        default=False,
                        action="store_true",
                        help="enable channels last for tensor cores")

    parser.add_argument('--log_dir',
                        default='/shared/export/logs',
                        type=str,
                        help='log directory path.')

    parser.add_argument('--label',
                        type=str,
                        default="resnet50_dev_delme",
                        help='label used to create log directory')

    # tensorboard files are pushed to S3 on validation (ideally infrequently)
    parser.add_argument('--bucket',
                        type=str,
                        default='mzanur-autoscaler',
                        help='s3 bucket for tensorboard')

    parser.add_argument("--checkpoint-file",
                        default="/shared/export/elastic/checkpoint.pth.tar",
                        type=str,
                        help="checkpoint file path, to load and save to")

    args = parser.parse_args()
    args.checkpoint_file = f'/shared/export/elastic/{args.label}/checkpoint.pth.tar'
    # if gradient accumulation file is found in S3 (written by autoscaler service,)
    # then use that file to update accumulation steps else use value passed in args
    # `grad_accum_change_detected` has side-effect of updating args 
    grad_accum_change_detected(args)
    return args


def setup_training(args):
    assert torch.cuda.is_available(), "Needs instance with GPU"
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        numpy.random.seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = True

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=600))
    args.rank = int(os.environ["RANK"])

    ########## AUTOSCALER SETUP ##########
    if args.enable_autoscaler:
        # FIXME: hardcoded
        args.autoscaler_cfg_path = "/gradstats/resnet50/imagenet/autoscaler.yaml"

    #NOTE: Rank ordering is not guaranteed across restarts
    args.logs_basedir = f'{args.log_dir}/{args.label}'
    args.tensorboard_path = f'{args.logs_basedir}/worker-{dist.get_rank()}'
    # create dirs that don't exist
    make_path_if_not_exists(args.tensorboard_path)
    return args


def main():
    args = parse_arguments()
    args = setup_training(args)
    main_worker(args)


# data pipeline enhancements from https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
def fast_collate(batch, memory_format):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros(
        (len(imgs), 3, h, w),
        dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if (nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets


def grad_accum_change_detected(args):
    # if gradient accumulation file is found in S3 (written by autoscaler service,)
    # then use that file to update accumulation steps else use value passed in args
    change_detected = False
    try:
        # FIXME: hardcoded for now 
        s3_prefix = f'resnet50/{args.label}/GNS/node_state'
        text = read_s3_textfile(args.bucket, s3_prefix)
        # read last line
        text = text.splitlines()[-1]
        num_workers, grad_accum_steps = [int(col) for col in text.split(',')]
        if args.gradient_accumulation_steps != grad_accum_steps:
            change_detected = True
        print(f'Setting gradient accumulation steps to {grad_accum_steps} as per recommendation')
        args.gradient_accumulation_steps = grad_accum_steps
    except Exception as e:
        print(e)
        print('Cannot find any recommendation from autoscaler service, continuing as before- current world size:', get_world_size())
    return change_detected


def main_worker(args):
    print("DDP training AMP enabled=", args.amp)
    # Always get device id from env for elastic 
    device_id = args.local_rank
    # loss scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # tensorboard summary writer (by default created for all workers)
    writer = SummaryWriter(args.tensorboard_path)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format
    model.cuda().to(memory_format=memory_format)

    model = DDP(model, device_ids=[args.gpu], output_device=args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer == "AdamW":
        # use own ADAMW (derived from PT optim)
        optimizer = AdamW(model.parameters(),
                                args.lr,
                                eps=args.eps,
                                betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay)

    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # wrap optimizer in AdaScale if predicting batch size or adjusting LR
    if args.enable_autoscaler:
        optimizer = AdaScale(
            optimizer,
            autoscaler_cfg_path=args.autoscaler_cfg_path,
            model=model,
            scaler=scaler,
            summary_writer=writer)
    else:
        optimizer.scale = 1
    torch.backends.cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]))

    collate_fn = lambda b: fast_collate(b, memory_format)

    if args.distributed:
        # seed should be time dependendent for elastic training
        SEED = int(100 * time.time() % 199)
        train_sampler = ReplacementDistributedSampler(train_dataset, seed=SEED)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               prefetch_factor=2,
                                               collate_fn=collate_fn)

    #FIXME: Check if still okay for elastic scenario
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             prefetch_factor=2,
                                             collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion, writer, epoch, args)
        writer.close()
        return

    # if we are using gradient accumulation then global_step will increment accum times per optimizer update
    args.print_freq = args.print_freq * args.gradient_accumulation_steps

    # for elastic we always resume from the latest checkpoint if one exists;
    state = load_checkpoint(args.checkpoint_file, device_id, args.arch, model, optimizer)

    start_epoch = state.epoch + 1
    global global_step
    global_step = state.global_step

    print(f"=====> start_epoch: {start_epoch}, best_acc1: {state.best_acc1}, restored_global_step: {global_step}")

    for epoch in range(start_epoch, args.epochs):
        state.epoch = epoch
        if grad_accum_change_detected(args):
            print("DETECTED CHANGE IN GRADIENT ACCUMULATION STEPS - RESTARTING CLUSTER")
            # simulate a failure here so that PT elastic relaunches cluster
            sys.exit(-1)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        if not args.run_gns_experiment:
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, writer, epoch, state, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, writer, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > state.best_acc1
        state.best_acc1 = max(acc1, state.best_acc1)

        if get_rank() == 0:
            save_checkpoint(state, is_best, args.checkpoint_file)
            if args.enable_autoscaler:
                # this fires a GNS info to be logged in S3.
                # Scaling service reads this info and initiates a resize if needed
                optimizer.check_for_cluster_resize()

    # close summary writer
    writer.close()


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255,
                                  0.406 * 255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255,
                                 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, model, criterion, optimizer, scaler, writer, epoch, state, args):
    global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
 
    effective_world_size = get_world_size() * args.gradient_accumulation_steps
    scale_one_gbs = int(args.batch_size * effective_world_size // optimizer.scale)
    scale_one_steps_per_epoch = int(len(train_loader) * args.batch_size // scale_one_gbs)

    print("==>", effective_world_size, len(train_loader), scale_one_gbs, scale_one_steps_per_epoch, global_step)

    progress = ProgressMeter(scale_one_steps_per_epoch,
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch (SI steps based): [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.perf_counter()

    prefetcher = data_prefetcher(train_loader)
    images, target = prefetcher.next()
    i = 0
    scheduler_progress = 0
    total_steps = 90 * scale_one_steps_per_epoch
    accumulate_gradients = args.gradient_accumulation_steps > 1
    curr_epoch_step = 0 # only to track grad accumulation related stuff
    while images is not None:
        global_step += 1
        state.global_step = global_step
        curr_epoch_step += 1
       
        ###### DEBUG ########
        # scale_one_steps_per_epoch = 100
        #####################
        # data scheme is sampling with replacement, scale_one_steps should be invariant for all scales
        if i >= scale_one_steps_per_epoch:
            break

        # measure data loading time
        data_time.update(time.perf_counter() - end)

        is_last_accumulation_step = curr_epoch_step % args.gradient_accumulation_steps == 0
        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            if not is_last_accumulation_step:
                with model.no_sync():
                    output = model(images)
                    #FIXME: this is per worker loss and for logging we may want to do average loss over world
                    loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss (on last batch of accumulation)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                average_factor = images.size(0) # * args.gradient_accumulation_steps
                losses.update(loss.item(), average_factor)
                top1.update(acc1[0], average_factor)
                top5.update(acc5[0], average_factor)

        if accumulate_gradients and not is_last_accumulation_step:
            with model.no_sync():
                scaler.scale(loss).backward()
        else:
            scaler.scale(loss).backward()
            # at the last accum step, take one optim step
            if args.enable_autoscaler:
                scheduler_progress = optimizer.get_step_increment()
                i += scheduler_progress
                optimizer.step()
            else:
                i = global_step % (scale_one_steps_per_epoch+1)
                scheduler_progress = 1
                scaler.step(optimizer)
            # update scaler state machine
            scaler.update()
            # optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            #torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # tensorboard summaries are logged based on scale invariant iterations
            # so that we can compare runs (loss values at the same logical stage)
            # NOTE if writing to S3 directly then make sure that write rate is limited else
            # S3 writes will fail and you will lose TB logs
            tensorboard_step = scale_one_steps_per_epoch * epoch + i
            if args.run_gns_experiment:
                # if running GNS experiments then adjust LR every step - instead of step decay
                linear_decay_learning_rate(optimizer, tensorboard_step, total_steps, args)

            if get_rank() == 0 and args.enable_autoscaler:
                optimizer.log_to_tensorboard(global_step // args.gradient_accumulation_steps)
            if curr_epoch_step % args.print_freq == 0 and get_rank() == 0:
                progress.display(i)
                writer.add_scalar('Train/Loss', losses.avg, tensorboard_step)
                writer.add_scalar('Train/Accuracy_top1', top1.avg, tensorboard_step)
                writer.add_scalar('Train/Accuracy_top5', top5.avg, tensorboard_step)
                writer.add_scalar('Train/Batch_time', batch_time.avg, tensorboard_step)
                writer.add_scalar('Train/Data_time', data_time.avg, tensorboard_step)
                if args.enable_autoscaler:
                    gain = optimizer.gain()
                    effective_lr = gain * optimizer.param_groups[0]['lr'] # assuming that all groups have same LR
                    gns = optimizer.gns()
                    print("gain={}\ngns={}\nsi_steps={}\neffective lr={}".format(gain, gns, scheduler_progress, effective_lr))
                # save checkpoint, flush and push to S3 every 500 iterations FIXME: hardcoded
                if global_step % 500 == 0:
                    save_checkpoint(state, False, args.checkpoint_file)
                    if args.enable_autoscaler:
                        optimizer.check_for_cluster_resize()
                    writer.flush()
        images, target = prefetcher.next()


def validate(val_loader, model, criterion, writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    end = time.perf_counter()

    prefetcher = data_prefetcher(val_loader)
    images, target = prefetcher.next()
    i = 0
    while images is not None:
        i += 1
        with torch.no_grad():
            # compute output
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(type(acc1), type(acc5[0]), acc1)
        # average across world size
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()

        if i % args.print_freq == 0:
            progress.display(i)
        images, target = prefetcher.next()

    # tensorboard update
    writer.add_scalar('Test/Accuracy_top1', top1.avg, epoch)
    writer.add_scalar('Test/Accuracy_top5', top5.avg, epoch)
    writer.flush()
    upload_dir(f'{args.logs_basedir}', args.bucket, f'{args.arch}/{args.label}')

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)


def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    arch: str,
    model: DDP,
    optimizer,
) -> State:
    state = State(arch, model, optimizer)
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")
    return state


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_decay_learning_rate(optimizer, step, total_steps, args):
    """ linear decay for GNS comparisons """
    base_lr = args.lr
    progress = step / total_steps
    lr = base_lr * (1.0 - progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
