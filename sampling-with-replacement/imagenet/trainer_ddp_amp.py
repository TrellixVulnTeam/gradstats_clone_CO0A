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
from torch.utils.tensorboard import SummaryWriter
from utils import upload_dir, make_path_if_not_exists

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

best_acc1 = 0


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

    parser.add_argument(
        '--enable_gns',
        default=False,
        action='store_true',
        help='Enable gradient noise scale measurement for training run')

    parser.add_argument('--enable_adascale',
                        default=False,
                        action='store_true',
                        help='Enable adascale module for training run')

    parser.add_argument('--gns_smoothing',
                        type=float,
                        default=0.0,
                        help='Smoothing factor for gradient stats.')

    parser.add_argument('--lr_scale',
                        type=float,
                        default=1.0,
                        help='Batch scaling factor for AdaScale.')

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

    parser.add_argument(
        '-b',
        '--batch-size',
        default=256,
        type=int,
        metavar='N',
        help='mini-batch size (default: 256), this is the total '
        'batch size of all GPUs when '
        'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.1,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')

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

    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--channels-last",
                        default=False,
                        action="store_true",
                        help="enable channels last for tensor cores")

    parser.add_argument('--log_dir',
                        default='/shared/logs',
                        type=str,
                        help='log directory path.')

    parser.add_argument('--label',
                        type=str,
                        default="resnet50_training",
                        help='label used to create log directory')

    parser.add_argument('--bucket',
                        type=str,
                        default='mzanur-autoscaler',
                        help='s3 bucket for tensorboard')

    args = parser.parse_args()
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
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.rank = int(os.environ["RANK"])
    args.tensorboard_path = f'{args.log_dir}/{args.label}/worker-{torch.distributed.get_rank()}'

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


def main_worker(args):
    global best_acc1
    print("DDP training AMP enabled=", args.amp)

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

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.gpu],
                                                      output_device=args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # wrap optimizer in AdaScale if predicting batch size or adjusting LR
    if args.enable_adascale or args.enable_gns:
        optimizer = AdaScale(
            optimizer,
            rank=get_rank(),
            is_adaptive=False,
            smoothing=None,  # smoothing coefficient determined by scale
            scaler=scaler,
            summary_writer=writer)
        optimizer.set_scale(args.lr_scale)
        print("=> adascale rank", get_rank(), "setting scale to",
              args.lr_scale)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

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
        train_sampler = ReplacementDistributedSampler(train_dataset,
                                                      seed=2047 * args.rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    # adjust batch size per worker
    args.batch_size = args.batch_size // get_world_size()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers,
                                               pin_memory=False,
                                               sampler=train_sampler,
                                               prefetch_factor=10,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False,
                                             sampler=val_sampler,
                                             prefetch_factor=10,
                                             collate_fn=collate_fn)

    if args.evaluate:
        validate(val_loader, model, criterion, writer, epoch, args)
        writer.close()
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, scaler, writer, epoch,
              args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, writer, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if get_rank() == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
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
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
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


global_step = 0  # this step has been added only for printing purposes, i.e. track progress every print_freq steps
def train(train_loader, model, criterion, optimizer, scaler, writer, epoch,
          args):
    global global_step
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    scale_one_bs = int(
        args.batch_size * get_world_size() // args.lr_scale
    )  # multiply by world size to account for division earlier
    scale_one_steps_per_epoch = int(
        len(train_loader) * args.batch_size // scale_one_bs)
    progress = ProgressMeter(scale_one_steps_per_epoch,
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()

    prefetcher = data_prefetcher(train_loader)
    images, target = prefetcher.next()
    i = 0
    adascale_step = 0
    print("====>", scale_one_steps_per_epoch, args.batch_size, get_world_size())
    while images is not None:
        global_step += 1
        # `i` below will be progressed as per Adascale gain
        # i += 1
        # Currently use ngpus_per_node, however this should be a dynamic value indicate the num_workers
        # which is also the data num_replicas
        if i >= scale_one_steps_per_epoch:
            break

        if get_rank() == 0:
            print("====>", i)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(
                output, target
            )  #FIXME: this is per worker loss and for logging we may want to do average loss over world

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        gns = 0.0
        gain = 1.0
        scaler.scale(loss).backward()
        if args.enable_gns or args.enable_adascale:
            if args.enable_gns:
                gns = optimizer.gns(scale_one_batch_size=scale_one_bs)
            if args.enable_adascale:
                gain = optimizer.scale_invariant_steps(
                    aggressive_base_schedule=False)
                prev_steps = math.floor(adascale_step)
                adascale_step = adascale_step + gain
                new_steps = math.floor(adascale_step)
                scale_invariant_steps = new_steps - prev_steps
                # progress base scale iteration `i` by scale_invariant_steps
                i += scale_invariant_steps
            # modify step according to gain
            optimizer.step()
        else:
            i = global_step % (scale_one_steps_per_epoch+1)
            scaler.step(optimizer)

        scaler.update()
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        #torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0 and get_rank() == 0:
            progress.display(i)
            print("=> gain {} gns {}".format(gain, gns))
            # tensorboard summaries are logged based on scale invariant iterations
            # so that we can compare runs (loss values at the same logical stage)
            tensorboard_step = scale_one_steps_per_epoch * epoch + i
            writer.add_scalar('Train/Real Iterations', global_step, tensorboard_step)
            writer.add_scalar('Train/Gain', gain, tensorboard_step)
            writer.add_scalar('Train/GNS', gns, tensorboard_step)
            writer.add_scalar('Train/Loss', losses.avg, tensorboard_step)
            writer.add_scalar('Train/Accuracy_top1', top1.avg, tensorboard_step)
            writer.add_scalar('Train/Accuracy_top5', top5.avg, tensorboard_step)
            writer.add_scalar('Train/Batch_time', batch_time.avg, tensorboard_step)
            writer.add_scalar('Train/Data_time', data_time.avg, tensorboard_step)
            # flush and push to S3 every 500 iterations FIXME: hardcoded
            if global_step % 500 == 0:
                writer.flush()
                # update the tensorboard log in s3 bucket
                upload_dir(f'{args.log_dir}/{args.label}', args.bucket,
                           f'{args.arch}/{args.label}')
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
    end = time.time()

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
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        images, target = prefetcher.next()
    # FIXME: REPORT average OVER ALL WORKERS

    # tensorboard update
    writer.add_scalar('Test/Accuracy_top1', top1.avg, epoch)
    writer.add_scalar('Test/Accuracy_top5', top5.avg, epoch)
    writer.flush()
    upload_dir(f'{args.log_dir}/{args.label}', args.bucket,
               f'{args.arch}/{args.label}')

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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

