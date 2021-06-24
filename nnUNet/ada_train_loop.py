from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.trainer.supporters import TensorRunningAccum, Accumulator

from copy import deepcopy

class AdaTrainLoop(TrainLoop):
    def on_train_epoch_start(self, epoch):

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        # re-init adascale step
        self.trainer.adascale_step = 0

        model = self.trainer.get_model()

        # reset train dataloader
        if self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # set seed for distributed sampler (enables shuffling for each epoch)
        try:
            self.trainer.train_dataloader.sampler.set_epoch(epoch)
        except Exception:
            pass

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_epoch_start(self.trainer, self.trainer.get_model())

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(
            window_length=self.trainer.accumulate_grad_batches
        )

        # structured result accumulators for callbacks
        self.early_stopping_accumulator = Accumulator()
        self.checkpoint_accumulator = Accumulator()

        # hook
        self.trainer.call_hook('on_epoch_start')
        self.trainer.call_hook('on_train_epoch_start')

    def run_training_epoch(self):

        # get model
        model = self.trainer.get_model()

        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator_backend.process_dataloader(self.trainer.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.num_optimizers)]

        # enable profiling for the dataloader
        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0
        should_check_val = False
        for batch_idx, (batch, is_last_batch) in train_dataloader:
            print(f"\nadascale_step is {self.trainer.adascale_step}\n"
                  f"scale_one_steps_per_epoch is {self.trainer.scale_one_steps_per_epoch}")
            if self.trainer.adascale_step > self.trainer.scale_one_steps_per_epoch:
                print('now we break!!!!!!!!!!')
                break
            self.trainer.batch_idx = batch_idx

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # only track outputs when user implements training_epoch_end
            # otherwise we will build up unnecessary memory
            epoch_end_outputs = self.process_train_step_outputs(
                batch_output.training_step_output_for_epoch_end,
                self.early_stopping_accumulator,
                self.checkpoint_accumulator
            )

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(epoch_output, epoch_end_outputs, batch, batch_idx, dataloader_idx)

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
            # -----------------------------------------
            should_check_val = self.should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.run_evaluation(test_mode=False)

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            monitor_metrics = deepcopy(self.trainer.logger_connector.callback_metrics)
            monitor_metrics.update(batch_output.batch_log_metrics)
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)

            # max steps reached, end training
            if self.trainer.max_steps is not None and self.trainer.max_steps == self.trainer.global_step + 1:
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.trainer.should_stop:
                break

            self.trainer.total_batch_idx += 1

            # stop epoch if we limited the number of training batches
            if batch_idx + 1 >= self.trainer.num_training_batches:
                break

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(
            epoch_output,
            self.checkpoint_accumulator,
            self.early_stopping_accumulator,
            self.num_optimizers
        )

        # hook
        self.trainer.logger_connector.on_train_epoch_end(epoch_output)

        # when no val loop is present or fast-dev-run still need to call checkpoints
        self.check_checkpoint_callback(not (should_check_val or is_overridden('validation_step', model)))

        # epoch end hook
        self.run_on_epoch_end_hook(epoch_output)

        # increment the global step once
        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()