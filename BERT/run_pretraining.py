# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
import multiprocessing

from tokenization import BertTokenizer
import modeling
from apex.optimizers import FusedLAMB, FusedAdam
from schedulers import PolyWarmUpScheduler

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from utils import is_main_process, format_step, get_world_size, get_rank, upload_dir
from torch.nn.parallel import DistributedDataParallel as DDP
from schedulers import LinearWarmUpScheduler

from automl.autoscaler import AdaScale
from torch.utils.tensorboard import SummaryWriter
import dllogger
from concurrent.futures import ProcessPoolExecutor

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

import signal


# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True


signal.signal(signal.SIGTERM, signal_handler)


#Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args,
                               worker_init):
    train_data = pretraining_dataset(input_file=input_file,
                                     max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu,
                                  num_workers=4,
                                  worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            'input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions',
            'masked_lm_ids', 'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):

        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else
            torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(
            as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [
            input_ids, segment_ids, input_mask, masked_lm_labels,
            next_sentence_labels
        ]


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(self, prediction_scores, seq_relationship_score,
                masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(
            prediction_scores.view(-1, self.vocab_size),
            masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2),
                                          next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


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





def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model",
                        default="bert-large-uncased",
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument( "--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                        "Sequences longer than this will be truncated, and sequences shorter \n"
                        "than this will be padded.")

    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")

    parser.add_argument("--use_adamw",
                        default=False,
                        action='store_true',
                        help="Use AdamW as the optimizer.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--adamw_beta1",
                        default=5e-5,
                        type=float,
                        help="Beta1 for Adam.")

    parser.add_argument("--adamw_beta2",
                        default=5e-5,
                        type=float,
                        help="Beta2 for Adam.")

    parser.add_argument("--adamw_weight_decay",
                        default=5e-5,
                        type=float,
                        help="Decoupled weight decay for Adam.")

    parser.add_argument("--adamw_eps",
                        default=5e-5,
                        type=float,
                        help="Epsilon for Adam.")

    parser.add_argument("--lr_poly_power",
                        default=1,
                        type=float,
                        help="Polynomial power for LR scheduler.")

    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")

    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")

    parser.add_argument('--amp',
                        default=False,
                        action='store_true',
                        help="Mixed precision training")

    parser.add_argument('--loss_scale',
                        type=float,
                        default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    parser.add_argument('--log_freq',
                        type=float,
                        default=1.0,
                        help='frequency of logging loss.')

    parser.add_argument('--checkpoint_activations',
                        default=False,
                        action='store_true',
                        help="Whether to use gradient checkpointing")

    # This should always be True for elastic training case
    parser.add_argument("--resume_from_checkpoint",
                        default=False,
                        action='store_true',
                        help="Whether to resume training from checkpoint.")

    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")

    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")

    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")

    parser.add_argument('--phase2',
                        default=False,
                        action='store_true',
                        help="Whether to train with seq len 512")

    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="Whether to do allreduces during gradient accumulation steps.")

    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=False,
                        action='store_true',
                        help="Whether to do fp16 allreduce post accumulation.")

    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=-1,
                        help="Number of training steps in Phase1 - seq len 128")

    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")

    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument('--json-summary',
                        type=str,
                        default="results/dllogger.json",
                        help='If provided, the json summary will be written to'
                        'the specified file.')

    parser.add_argument("--use_env",
                        action='store_true',
                        help="Whether to read local rank from ENVVAR")

    parser.add_argument('--disable_progress_bar',
                        default=False,
                        action='store_true',
                        help='Disable tqdm progress bar')

    parser.add_argument('--steps_this_run',
                        type=int,
                        default=-1,
                        help='If provided, only run this many steps before exiting')

    parser.add_argument('--log_dir',
                        default='/home/ubuntu/workspace/logs',
                        type=str,
                        help='log directory path.')

    parser.add_argument('--label',
                        type=str,
                        default="bert_training",
                        help='label used to create log directory')

    parser.add_argument('--bucket',
                        type=str,
                        default='mzanur-autoscaler',
                        help='s3 bucket for tensorboard')

    parser.add_argument('--autoscaler-cfg-path',
                        default="/home/ubuntu/workspace/gradstats/BERT/autoscaler.yaml",
                        type=str,
                        help='AutoScaler configuration')

    parser.add_argument('--enable-autoscaler',
                        default=False,
                        action='store_true',
                        help='when enabled we start measuring gradient stats')

    parser.add_argument('--sampling_with_replacement',
                        default=False,
                        action='store_true',
                        help='Each rank sees a different shuffle of full dataset - if enabled then sharding is not done')


    args = parser.parse_args()
    args.fp16 = args.fp16 or args.amp

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    return args


def setup_training(args):

    assert (torch.cuda.is_available())

    if args.local_rank == -1:
        device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.n_gpu = 1

    if args.gradient_accumulation_steps == 1:
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False

    if is_main_process():
        dllogger.init(backends=[
            dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE, filename=args.json_summary),
            dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE, step_format=format_step)
        ])
    else:
        dllogger.init(backends=[])

    print(f"device: {device} n_gpu: {args.n_gpu}, distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}")

    if args.gradient_accumulation_steps < 1:
        raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, batch size {args.train_batch_size} should be divisible")

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train:
        raise ValueError(" `do_train`  must be True.")

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and \
            (os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args


def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if f.endswith(".pt")]
            if len(model_names) > 0:
                args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])
            else:
                print('No checkpoint found, resetting step to 0')
                args.resume_step = 0
                args.resume_from_checkpoint = False

        global_step = args.resume_step if not args.init_checkpoint else 0

        if args.resume_from_checkpoint: # a valid checkpoint was found
            # load latest checkpoint
            if not args.init_checkpoint:
                ckpt_path = (os.path.join(args.output_dir, f"ckpt_{global_step}.pt"))
            else:
                ckpt_path = args.init_checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
            print("checkpoint keys ", checkpoint.keys())

            # differentiate between phase1 and phase2 restart
            is_this_first_phase2_run = False
            if checkpoint['phase'] == 1 and args.phase2:
                is_this_first_phase2_run = True

            if args.phase2:
                if not args.init_checkpoint and args.phase1_end_step != -1:
                    global_step -= args.phase1_end_step
                elif is_this_first_phase2_run:
                    # with AdaScale we don't have a known value for phase1 end step we reset global step to 0
                    global_step = 0
                    args.phase1_end_step = args.resume_step
                else:
                    # here we are resuming a phase2 training
                    args.phase1_end_step = checkpoint['phase1_end_step']
                    global_step = args.resume_step - args.phase1_end_step

        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    if args.use_adamw:
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.adamw_weight_decay},
            {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}]
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(args.adamw_beta1, args.adamw_beta2),
                              eps=args.adamw_eps,
                              adam_w_mode=True)
    else:
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01},
            {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}]
        optimizer = FusedLAMB(optimizer_grouped_parameters, lr=args.learning_rate)

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    args.tensorboard_path = f'{args.log_dir}/{args.label}/worker-{torch.distributed.get_rank()}'
    writer = SummaryWriter(args.tensorboard_path)

    if args.enable_autoscaler:
        optimizer = AdaScale(
            optimizer,
            autoscaler_cfg_path=args.autoscaler_cfg_path,
            num_grads_to_accum=args.gradient_accumulation_steps,
            model=model,
            scaler=scaler,
            summary_writer=writer)
    else:
        optimizer.scale = 1

    lr_scheduler = PolyWarmUpScheduler(
        optimizer,
        warmup=args.warmup_proportion,
        total_steps=args.max_steps,
        degree=args.lr_poly_power if args.use_adamw else 0.5,
        do_poly_warmup=True if args.use_adamw else False)

    model.checkpoint_activations(args.checkpoint_activations)

    ###############################################################################
    #TODO: Migrate checkpointing functionality to AutoScaler checkpoint State class
    ###############################################################################
    args.scale_invariant_steps = 0
    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for pg_idx, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][pg_idx]['step'] = global_step
                checkpoint['optimizer']['param_groups'][pg_idx]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][pg_idx]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][pg_idx]['lr'] = args.learning_rate
        if (not args.phase2) or (not is_this_first_phase2_run):
            lr_scheduler.last_epoch = checkpoint['optimizer']['state']['adascale']['scale_invariant_steps']
        else:
            lr_scheduler.last_epoch = 0
        args.scale_invariant_steps = lr_scheduler.last_epoch

        # adjust scale invariant steps in the checkpoint (if we reset it to zero for phase 2)
        checkpoint['optimizer']['state']['adascale']['scale_invariant_steps'] = args.scale_invariant_steps
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Restore AMP master parameters
        if args.fp16:
            scaler.load_state_dict(checkpoint['scaler'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    ###############################################################################

    if args.local_rank != -1:
        if args.allreduce_post_accumulation:
            # AutoScaler change - use standard DDP instead of Apex DDP
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        else:
            raise NotImplementedError
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion, scaler, writer


def take_optimizer_step(args, scaler, optimizer, model, global_step):
    if args.allreduce_post_accumulation:
        if args.enable_autoscaler:
            # optimizer is adascale wrapped and we pass scaler as an argument to get loss scale
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
        # update scaler state machine
        scaler.update()
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1
    else:
        raise NotImplementedError
    return global_step


def main():
    global timeout_sent

    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)

    device, args = setup_training(args)
    dllogger.log(step="PARAMETER", data={"Config": [str(args)]})

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, criterion, scaler, writer = prepare_model_and_optimizer(args, device)

    if is_main_process():
        dllogger.log(step="PARAMETER", data={"SEED": args.seed})

    raw_train_start = None
    if args.do_train:
        if is_main_process():
            dllogger.log(step="PARAMETER", data={"train_start": True})
            dllogger.log(step="PARAMETER",
                         data={"batch_size_per_gpu": args.train_batch_size})
            dllogger.log(step="PARAMETER",
                         data={"learning_rate": args.learning_rate})

        model.train()
        most_recent_ckpts_paths = []
        average_loss = 0.0  # averaged loss every args.log_freq steps
        epoch = 0
        training_steps = 0
        adascale_step = args.scale_invariant_steps
        accumulate_gradients = args.gradient_accumulation_steps > 1

        pool = ProcessPoolExecutor(1)

        # Note: We loop infinitely over epochs, termination is handled via iteration count
        while True:
            thread = None
            restored_data_loader = None
            if not args.resume_from_checkpoint or epoch > 0 or \
                    (args.phase2 and global_step < 1) or \
                    args.init_checkpoint:
                files = [
                    os.path.join(args.input_dir, f)
                    for f in os.listdir(args.input_dir)
                    if os.path.isfile(os.path.join(args.input_dir, f))
                    and 'training' in f
                ]
                files.sort()
                num_files = len(files)
                if args.sampling_with_replacement:
                    # different shuffle per rank per epoch (since we have multiple epochs for wiki+books)
                    random.Random(args.seed + epoch + get_rank()).shuffle(files)
                else:
                    random.Random(args.seed + epoch).shuffle(files)
                f_start_id = 0
            else:
                f_start_id = checkpoint['files'][0]
                files = checkpoint['files'][1:]
                args.resume_from_checkpoint = False
                num_files = len(files)
                # may not exist in all checkpoints
                epoch = checkpoint.get('epoch', 0)
                restored_data_loader = checkpoint.get('data_loader', None)
                # if sampling with replacement shuffle files so that workers start from different points
                random.Random(args.seed + epoch + get_rank()).shuffle(files)

            shared_file_list = {}

            if torch.distributed.is_initialized() and get_world_size() > num_files:
                remainder = get_world_size() % num_files
                data_file = files[(f_start_id * get_world_size() + get_rank() + remainder * f_start_id) % num_files]
            elif args.sampling_with_replacement:
                data_file = files[f_start_id]
                print("worker", get_rank(), data_file)
            else:
                data_file = files[(f_start_id * get_world_size() + get_rank()) % num_files]

            previous_file = data_file

            if restored_data_loader is None:
                train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
                train_sampler = RandomSampler(train_data)
                train_dataloader = DataLoader(train_data,
                                    sampler=train_sampler,
                                    batch_size=args.train_batch_size * args.n_gpu,
                                    num_workers=12,
                                    worker_init_fn=worker_init,
                                    pin_memory=True)
                # shared_file_list["0"] = (train_dataloader, data_file)
            else:
                train_dataloader = restored_data_loader
                restored_data_loader = None

            for f_id in range(f_start_id + 1, len(files)):

                if get_world_size() > num_files:
                    data_file = files[(f_id * get_world_size() + get_rank() + remainder * f_id) % num_files]
                else:
                    data_file = files[(f_id * get_world_size() + get_rank()) % num_files]

                # overwrite data file to look at current index
                if args.sampling_with_replacement:
                    data_file = files[f_id]
                print("inside loop worker", get_rank(), data_file)

                previous_file = data_file

                dataset_future = pool.submit(create_pretraining_dataset,
                                             data_file,
                                             args.max_predictions_per_seq,
                                             shared_file_list, args,
                                             worker_init)

                train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) \
                        if is_main_process() else train_dataloader

                if raw_train_start is None:
                    raw_train_start = time.time()

                for step, batch in enumerate(train_iter):  # produce batch per gpu
                    training_steps += 1
                    is_last_accumulation_step = training_steps % args.gradient_accumulation_steps == 0
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                    with torch.cuda.amp.autocast(enabled=args.fp16):
                        if not is_last_accumulation_step:
                            with model.no_sync():
                                prediction_scores, seq_relationship_score = model(input_ids=input_ids,
                                                                                token_type_ids=segment_ids,
                                                                                attention_mask=input_mask)
                                loss = criterion(prediction_scores,
                                                 seq_relationship_score,
                                                 masked_lm_labels,
                                                 next_sentence_labels)
                        else:
                            prediction_scores, seq_relationship_score = model(input_ids=input_ids,
                                                                            token_type_ids=segment_ids,
                                                                            attention_mask=input_mask)
                            loss = criterion(prediction_scores,
                                             seq_relationship_score,
                                             masked_lm_labels,
                                             next_sentence_labels)
                        if args.n_gpu > 1:
                            loss = loss.mean()  # DataParallel case

                    divisor = args.gradient_accumulation_steps
                    if accumulate_gradients:
                        if not args.allreduce_post_accumulation:
                            # this division was merged into predivision
                            loss = loss / args.gradient_accumulation_steps
                            divisor = 1.0
                    if accumulate_gradients and not is_last_accumulation_step:
                        with model.no_sync():
                            # for this to work correctly ensure that loss calc is in similar context
                            scaler.scale(loss).backward()
                    else:
                        scaler.scale(loss).backward()
                    average_loss += loss.item()

                    # take one optimizer step for gradient accumulation steps
                    if training_steps % args.gradient_accumulation_steps == 0:
                        if args.enable_autoscaler:
                            scheduler_progress = optimizer.get_step_increment()
                            adascale_step += scheduler_progress
                            lr_scheduler.step(step_increment=scheduler_progress)
                        else:
                            lr_scheduler.step()
                        global_step = take_optimizer_step(args, scaler, optimizer, model, global_step)

                    learning_rate = optimizer.param_groups[0]['lr']
                    if adascale_step >= args.steps_this_run or timeout_sent:
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = average_loss.item()
                        if is_main_process():
                            dllogger.log(step=(epoch, global_step,), data={"final_loss": final_loss})
                        adascale_step += 1
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        average_loss /= (args.log_freq * divisor)
                        if args.enable_autoscaler:
                            gain = optimizer.gain()
                            gns = optimizer.gns()
                        else:
                            gain = 1.0
                            gns = 0
                            adascale_step = global_step

                        if is_main_process():
                            dllogger.log(step=(epoch, global_step,),
                                data={
                                    "average_loss": average_loss,
                                    "step_loss": loss.item() * args.gradient_accumulation_steps / divisor,
                                    "learning_rate": learning_rate,
                                    "gain": gain,
                                    "gns": gns,
                                    "effective_lr": learning_rate * gain,
                                    "scale_invariant_steps": adascale_step
                                })
                        # for all workers log tensorboard summary
                        phase = 2 if args.phase2 else 1
                        if get_rank() == 0:
                            writer.add_scalar(f'Train{phase}/Loss', average_loss, adascale_step)
                            if args.enable_autoscaler:
                                optimizer.log_to_tensorboard(adascale_step, phase)
                            writer.flush()
                        # pushing to S3 is a sync call at the moment and is very expensive so we reduce the frequency of push
                        if training_steps % 10 == 0:
                            # update the tensorboard log in s3 bucket
                            res = upload_dir(f'{args.log_dir}/{args.label}', args.bucket, f'BERT/{args.label}')
                            if not res:
                                print("Failed to push to S3")
                        # reset average loss for next print loop
                        average_loss = 0
                    if adascale_step > args.steps_this_run or \
                            training_steps % (args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0 or \
                            timeout_sent:
                        if is_main_process() and not args.skip_checkpoint:
                            # Save a trained model
                            dllogger.log(step="PARAMETER", data={"checkpoint_step": global_step})
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, f"ckpt_{global_step}.pt")
                            else:
                                output_save_file = os.path.join(args.output_dir,f"ckpt_{global_step+args.phase1_end_step}.pt")
                            if args.do_train:
                                #TODO: Migrate to new ckpt class for AutoScaler
                                torch.save({'model': model_to_save.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'scaler': scaler.state_dict(),
                                            'files': [f_id] + files,
                                            'epoch': epoch,
                                            'data_loader': None if adascale_step >= args.max_steps else train_dataloader,
                                            'phase': 2 if args.phase2 else 1, # using this to differentiate between phase1 and phase2 restarts
                                            'phase1_end_step': args.phase1_end_step,}, output_save_file)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 30:
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed)

                        # Exiting the training due to hitting max steps, or being sent a
                        # timeout from the cluster scheduler
                        if adascale_step > args.steps_this_run or timeout_sent:
                            del train_dataloader
                            writer.close()
                            return args, final_loss, train_time_raw, global_step

                del train_dataloader
                # Make sure pool has finished and switch train_dataloader
                # NOTE: Will block until complete
                train_dataloader, data_file = dataset_future.result(timeout=None)
            epoch += 1
    writer.close()


if __name__ == "__main__":

    now = time.time()
    args, final_loss, train_time_raw, global_step = main()
    gpu_count = args.n_gpu
    global_step += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
    if args.resume_step == -1:
        args.resume_step = 0
    if torch.distributed.is_initialized():
        gpu_count = get_world_size()
    if is_main_process():
        e2e_time = time.time() - now
        training_perf = args.train_batch_size * args.gradient_accumulation_steps * gpu_count\
                        * (global_step - args.resume_step + skipped_steps) / train_time_raw
        dllogger.log(step=tuple(),
                     data={
                         "e2e_train_time": e2e_time,
                         "training_sequences_per_second": training_perf,
                         "final_loss": final_loss,
                         "raw_train_time": train_time_raw
                     })
    dllogger.flush()

