from pytorch_lightning import Trainer
from ada_train_loop import AdaTrainLoop

import os
import warnings
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.step_result import EvalResult
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.trainer.callback_hook import TrainerCallbackHookMixin
from pytorch_lightning.trainer.configuration_validator import ConfigValidator
from pytorch_lightning.trainer.connectors.env_vars_connector import overwrite_by_env_vars
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from pytorch_lightning.trainer.logging import TrainerLoggingMixin
from pytorch_lightning.trainer.model_hooks import TrainerModelHooksMixin
from pytorch_lightning.trainer.optimizers import TrainerOptimizersMixin
from pytorch_lightning.trainer.states import TrainerState, trainer_state
from pytorch_lightning.trainer.training_tricks import TrainerTrainingTricksMixin
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.debugging import InternalDebugger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.trainer.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.accelerators.accelerator_connector import AcceleratorConnector
from pytorch_lightning.trainer.connectors.logger_connector import LoggerConnector
from pytorch_lightning.trainer.connectors.optimizer_connector import OptimizerConnector
from pytorch_lightning.trainer.connectors.training_trick_connector import TrainingTricksConnector
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector
from pytorch_lightning.trainer.connectors.model_connector import ModelConnector
from pytorch_lightning.trainer.connectors.debugging_connector import DebuggingConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
from pytorch_lightning.trainer.connectors.slurm_connector import SLURMConnector
from pytorch_lightning import _logger as log
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.trainer.connectors.precision_connector import PrecisionConnector
from pytorch_lightning.trainer.connectors.profiler_connector import ProfilerConnector
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer.properties import TrainerProperties
from pytorch_lightning.plugins.plugin_connector import PluginConnector
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu_accelerator import CPUAccelerator

class AdaTrainer(Trainer):
    @overwrite_by_env_vars
    def __init__(
        self,
        logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] = True,
        checkpoint_callback: Union[ModelCheckpoint, bool] = True,
        callbacks: Optional[List[Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0,
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: int = 1,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: bool = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: int = 1000,
        min_epochs: int = 1,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 50,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None,
        profiler: Optional[Union[BaseProfiler, bool]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: list = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        automatic_optimization: bool = True,
        train_dataloader_len=None,
        args = None
    ):
        r"""
        Customize every aspect of training via flags
        Args:
            accelerator: Previously known as distributed_backend (dp, ddp, ddp2, etc...).
                Can also take in an accelerator object for custom hardware.
            accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.
            amp_backend: The mixed precision backend to use ("native" or "apex")
            amp_level: The optimization level to use (O1, O2, etc...).
            auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
                trying to optimize initial learning for faster convergence. trainer.tune() method will
                set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                To use a different key set a string instead of True with the key name.
            auto_scale_batch_size: If set to True, will `initially` run a batch size
                finder trying to find the largest batch size that fits into memory.
                The result will be stored in self.batch_size in the LightningModule.
                Additionally, can be set to either `power` that estimates the batch size through
                a power search or `binsearch` that estimates the batch size through a binary search.
            auto_select_gpus: If enabled and `gpus` is an integer, pick available
                gpus automatically. This is especially useful when
                GPUs are configured to be in "exclusive mode", such
                that only one process at a time can access them.
            benchmark: If true enables cudnn.benchmark.
            callbacks: Add a list of callbacks.
            checkpoint_callback: Callback for checkpointing.
            check_val_every_n_epoch: Check val every n train epochs.
            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Default: ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            deterministic: If true enables cudnn.deterministic.
            distributed_backend: deprecated. Please use 'accelerator'
            fast_dev_run: runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).
            flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).
            gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
            gradient_clip_val: 0 means don't clip.
            limit_train_batches: How much of training dataset to check (floats = percent, int = num_batches)
            limit_val_batches: How much of validation dataset to check (floats = percent, int = num_batches)
            limit_test_batches: How much of test dataset to check (floats = percent, int = num_batches)
            logger: Logger (or iterable collection of loggers) for experiment tracking.
            log_gpu_memory: None, 'min_max', 'all'. Might slow performance
            log_every_n_steps: How often to log within steps (defaults to every 50 steps).
            automatic_optimization: If False you are responsible for calling .backward, .step, zero_grad.
                Meant to be used with multiple optimizers by advanced users.
            prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
            process_position: orders the progress bar when running multiple models on same machine.
            progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
                Ignored when a custom callback is passed to :paramref:`~Trainer.callbacks`.
            profiler:  To profile individual steps during training and assist in identifying bottlenecks.
            overfit_batches: Overfit a percent of training data (float) or a set number of batches (int). Default: 0.0
            plugins: Plugins allow modification of core behavior like ddp and amp.
            precision: Full precision (32), half precision (16). Can be used on CPU, GPU or TPUs.
            max_epochs: Stop training once this number of epochs is reached.
            min_epochs: Force training for at least these many epochs
            max_steps: Stop training after this number of steps. Disabled by default (None).
            min_steps: Force training for at least these number of steps. Disabled by default (None).
            num_nodes: number of GPU nodes for distributed training.
            num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
                Set it to `-1` to run all batches in all validation dataloaders. Default: 2
            reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.
            replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
                will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
                train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
                you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
            resume_from_checkpoint: To resume training from a specific checkpoint pass in the path here.
                This can be a URL.
            sync_batchnorm: Synchronize batch norm layers between process groups/whole world.
            terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
                end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
            tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]
            track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.
            truncated_bptt_steps: Truncated back prop breaks performs backprop every k steps of much longer
                sequence.
            val_check_interval: How often to check the validation set. Use float to check within a training epoch,
                use int to check every n steps (batches).
            weights_summary: Prints a summary of the weights when training begins.
            weights_save_path: Where to save weights if specified. Will override default_root_dir
                    for checkpoints only. Use this if for whatever reason you need the checkpoints
                    stored in a different place than the logs written in `default_root_dir`.
                    Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                    Defaults to `default_root_dir`.
        """
        # super().__init__()

        # init connectors
        self.dev_debugger = InternalDebugger(self)
        self.config_validator = ConfigValidator(self)
        self.data_connector = DataConnector(self)
        self.optimizer_connector = OptimizerConnector(self)
        self.accelerator_connector = AcceleratorConnector(self)
        self.logger_connector = LoggerConnector(self)
        self.model_connector = ModelConnector(self)
        self.precision_connector = PrecisionConnector(self)
        self.callback_connector = CallbackConnector(self)
        self.debugging_connector = DebuggingConnector(self)
        self.training_tricks_connector = TrainingTricksConnector(self)
        self.profile_connector = ProfilerConnector(self)
        self.checkpoint_connector = CheckpointConnector(self)
        self.slurm_connector = SLURMConnector(self)
        self.tuner = Tuner(self)
        self.accelerator_backend = None
        self.evaluation_loop = EvaluationLoop(self)
        self.train_loop = AdaTrainLoop(self)
        self.plugin_connector = PluginConnector(self)
        # training state
        self.weights_summary = weights_summary
        self.model = None
        self.shown_warnings = set()

        # init adascale related
        self.adascale_step = None # adascale normal step
        self.adascale_accu_step = None # adascale accumulation step
        self.scale_one_bs = None
        self.scale_one_steps_per_epoch = None

        # args
        self.args = args

        # writer
        self.writer = None

        # init callbacks
        # Declare attributes to be set in callback_connector on_trainer_init
        self.checkpoint_callback: Union[ModelCheckpoint, bool] = checkpoint_callback
        self.callback_connector.on_trainer_init(
            callbacks,
            checkpoint_callback,
            progress_bar_refresh_rate,
            process_position,
            default_root_dir,
            weights_save_path,
            resume_from_checkpoint,
        )

        # hook
        self.on_init_start()

        # init optimizer + lr scheduler related flags
        self.optimizer_connector.on_trainer_init()

        # init data flags
        self.data_connector.on_trainer_init(
            check_val_every_n_epoch, reload_dataloaders_every_epoch, prepare_data_per_node
        )

        # init training tricks
        self.training_tricks_connector.on_trainer_init(
            gradient_clip_val, track_grad_norm, accumulate_grad_batches, truncated_bptt_steps, terminate_on_nan
        )

        # init accelerator related flags
        self.accelerator_connector.on_trainer_init(
            num_processes,
            tpu_cores,
            accelerator,
            distributed_backend,
            auto_select_gpus,
            gpus,
            num_nodes,
            log_gpu_memory,
            sync_batchnorm,
            benchmark,
            replace_sampler_ddp,
            deterministic,
        )

        # init train loop related flags
        self.train_loop.on_trainer_init(
            max_epochs,
            min_epochs,
            max_steps,
            min_steps,
            num_sanity_val_steps,
            automatic_optimization
        )
        self.evaluation_loop.on_trainer_init()

        # configure tuner
        self.tuner.on_trainer_init(auto_lr_find, auto_scale_batch_size)

        # configure profiler
        self.profile_connector.on_trainer_init(profiler)

        # init logger flags
        self.logger_connector.on_trainer_init(logger, flush_logs_every_n_steps, log_every_n_steps)

        # init debugging flags
        self.debugging_connector.on_init_start(
            limit_train_batches,
            limit_val_batches,
            limit_test_batches,
            val_check_interval,
            overfit_batches,
            fast_dev_run,
        )

        # set precision
        self.precision_connector.on_trainer_init(precision, amp_level, amp_backend)

        # last thing are the plugins which override whatever the trainer used by default
        self.plugin_connector.on_trainer_init(plugins)

        # Callback system
        self.on_init_end()

