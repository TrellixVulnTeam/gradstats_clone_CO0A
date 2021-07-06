# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
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

import os

import math
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from fairscale.optim import AdaScale
from apex.optimizers import FusedAdam, FusedSGD
from monai.inferers import sliding_window_inference
from skimage.transform import resize
from torch_optimizer import RAdam
from utils.utils import (
    flip,
    get_dllogger,
    get_path,
    get_test_fnames,
    get_tta_flips,
    get_unet_params,
    is_main_process,
    layout_2d,
)

from models.loss import Loss
from models.metrics import Dice
from models.unet import UNet

class NNUnet(pl.LightningModule):
    def __init__(self, args, bermuda=False, data_dir=None):
        super(NNUnet, self).__init__()
        self.args = args
        self.bermuda = bermuda
        if data_dir is not None:
            self.args.data = data_dir
        self.save_hyperparameters()
        self.build_nnunet()
        self.best_sum = 0
        self.best_sum_epoch = 0
        self.best_dice = self.n_class * [0]
        self.best_epoch = self.n_class * [0]
        self.best_sum_dice = self.n_class * [0]
        self.test_idx = 0
        self.test_imgs = []
        self.cur_loss = None
        if not self.bermuda:
            self.learning_rate = args.learning_rate
            self.loss = Loss(self.args.focal)
            self.tta_flips = get_tta_flips(args.dim)
            self.dice = Dice(self.n_class)
            if self.args.exec_mode in ["train", "evaluate"]:
                self.dllogger = get_dllogger(args.results)

        self.gns = 0.0
        self.gain = 1.0

    def forward(self, img):
        return torch.argmax(self.model(img), 1)

    def _forward(self, img):
        if self.args.benchmark:
            if self.args.dim == 2 and self.args.data2d_dim == 3:
                img = layout_2d(img, None)
            return self.model(img)
        return self.tta_inference(img) if self.args.tta else self.do_inference(img)

    def training_step(self, batch, batch_idx):
        img, lbl = self.get_train_data(batch)
        pred = self.model(img)
        loss = self.loss(pred, lbl)
        self.cur_loss = loss
        return loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.args.skip_first_n_eval:
            return None
        img, lbl = batch["image"], batch["label"]
        pred = self._forward(img)
        loss = self.loss(pred, lbl)
        self.dice.update(pred, lbl[:, 0])
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        if self.args.exec_mode == "evaluate":
            return self.validation_step(batch, batch_idx)
        img = batch["image"]
        pred = self._forward(img)
        if self.args.save_preds:
            meta = batch["meta"][0].cpu().detach().numpy()
            original_shape = meta[2]
            min_d, max_d = meta[0, 0], meta[1, 0]
            min_h, max_h = meta[0, 1], meta[1, 1]
            min_w, max_w = meta[0, 2], meta[1, 2]

            final_pred = torch.zeros((1, pred.shape[1], *original_shape), device=img.device)
            final_pred[:, :, min_d:max_d, min_h:max_h, min_w:max_w] = pred
            final_pred = nn.functional.softmax(final_pred, dim=1)
            final_pred = final_pred.squeeze(0).cpu().detach().numpy()

            if not all(original_shape == final_pred.shape[1:]):
                class_ = final_pred.shape[0]
                resized_pred = np.zeros((class_, *original_shape))
                for i in range(class_):
                    resized_pred[i] = resize(
                        final_pred[i], original_shape, order=3, mode="edge", cval=0, clip=True, anti_aliasing=False
                    )
                final_pred = resized_pred

            self.save_mask(final_pred)

    def build_nnunet(self):
        in_channels, n_class, kernels, strides, self.patch_size = get_unet_params(self.args)
        self.n_class = n_class - 1
        self.model = UNet(
            in_channels=in_channels,
            n_class=n_class,
            kernels=kernels,
            strides=strides,
            dimension=self.args.dim,
            residual=self.args.residual,
            normalization_layer=self.args.norm,
            negative_slope=self.args.negative_slope,
        )
        if is_main_process():
            print(f"Filters: {self.model.filters},\nKernels: {kernels}\nStrides: {strides}")

    def do_inference(self, image):
        if self.args.dim == 3:
            return self.sliding_window_inference(image)
        if self.args.data2d_dim == 2:
            return self.model(image)
        if self.args.exec_mode == "predict":
            return self.inference2d_test(image)
        return self.inference2d(image)

    def tta_inference(self, img):
        pred = self.do_inference(img)
        for flip_idx in self.tta_flips:
            pred += flip(self.do_inference(flip(img, flip_idx)), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def inference2d(self, image):
        batch_modulo = image.shape[2] % self.args.val_batch_size
        if batch_modulo != 0:
            batch_pad = self.args.val_batch_size - batch_modulo
            image = nn.ConstantPad3d((0, 0, 0, 0, batch_pad, 0), 0)(image)
        image = torch.transpose(image.squeeze(0), 0, 1)
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for start in range(0, image.shape[0] - self.args.val_batch_size + 1, self.args.val_batch_size):
            end = start + self.args.val_batch_size
            pred = self.model(image[start:end])
            preds[start:end] = pred.data
        if batch_modulo != 0:
            preds = preds[batch_pad:]
        return torch.transpose(preds, 0, 1).unsqueeze(0)

    def inference2d_test(self, image):
        preds_shape = (image.shape[0], self.n_class + 1, *image.shape[2:])
        preds = torch.zeros(preds_shape, dtype=image.dtype, device=image.device)
        for depth in range(image.shape[2]):
            preds[:, :, depth] = self.sliding_window_inference(image[:, :, depth])
        return preds

    def sliding_window_inference(self, image):
        return sliding_window_inference(
            inputs=image,
            roi_size=self.patch_size,
            sw_batch_size=self.args.val_batch_size,
            predictor=self.model,
            overlap=self.args.overlap,
            mode=self.args.blend,
        )

    @staticmethod
    def metric_mean(name, outputs):
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

    def validation_epoch_end(self, outputs):
        if self.current_epoch < self.args.skip_first_n_eval:
            self.log("dice_sum", 0.001 * self.current_epoch)
            self.dice.reset()
            return None
        loss = self.metric_mean("val_loss", outputs)
        dice = self.dice.compute()
        dice_sum = torch.sum(dice)
        if dice_sum >= self.best_sum:
            self.best_sum = dice_sum
            self.best_sum_dice = dice[:]
            self.best_sum_epoch = self.current_epoch
        for i, dice_i in enumerate(dice):
            if dice_i > self.best_dice[i]:
                self.best_dice[i], self.best_epoch[i] = dice_i, self.current_epoch

        if is_main_process():
            metrics = {}
            mean_dice = round(torch.mean(dice).item(), 2)
            TOP_mean = round(torch.mean(self.best_sum_dice).item(), 2)
            metrics.update({"mean dice": mean_dice})
            metrics.update({"TOP_mean": TOP_mean})
            if self.n_class > 1:
                metrics.update({f"L{i+1}": round(m.item(), 2) for i, m in enumerate(dice)})
                metrics.update({f"TOP_L{i+1}": round(m.item(), 2) for i, m in enumerate(self.best_sum_dice)})
            val_loss = round(loss.item(), 4)
            metrics.update({"val_loss": val_loss})
            self.dllogger.log(step=self.current_epoch, data=metrics)
            self.dllogger.flush()

            # tensorboard writer
            self.trainer.writer.add_scalar('Test/Mean Dice', mean_dice, self.trainer.current_epoch)
            self.trainer.writer.add_scalar('Test/Top Mean', TOP_mean, self.trainer.current_epoch)
            self.trainer.writer.add_scalar('Test/Val Loss', val_loss, self.trainer.current_epoch)
        self.log("val_loss", loss)
        self.log("dice_sum", dice_sum)



    def test_epoch_end(self, outputs):
        if self.args.exec_mode == "evaluate":
            self.eval_dice = self.dice.compute()

    def configure_optimizers(self):
        optimizer = {
            "sgd": FusedSGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum),
            "adam": FusedAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
            "radam": RAdam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        import torch.distributed as dist
        def get_rank():
            if not dist.is_available():
                return 0
            if not dist.is_initialized():
                return 0
            return dist.get_rank()

        # wrap optimizer in AdaScale if predicting batch size or adjusting LR
        if self.args.enable_adascale or self.args.enable_gns:
            optimizer = AdaScale(
                optimizer,
                rank=get_rank(),
                is_adaptive=True,
                smoothing=None,  # smoothing coefficient determined by scale
                trainer=self.trainer)
            optimizer.set_scale(self.args.lr_scale)

        scheduler = {
            "none": None,
            "multistep": torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.steps, gamma=self.args.factor),
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_epochs),
            "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.factor, patience=self.args.lr_patience
            ),
        }[self.args.scheduler.lower()]

        opt_dict = {"optimizer": optimizer, "monitor": "val_loss"}
        if scheduler is not None:
            opt_dict.update({"lr_scheduler": scheduler})
        return opt_dict

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        second_order_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        r"""
        Override this method to adjust the default way the
        :class:`~pytorch_lightning.trainer.trainer.Trainer` calls each optimizer.
        By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example
        once per optimizer.
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            second_order_closure: closure for second order methods
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        Examples:
            .. code-block:: python
                # DEFAULT
                def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                                   second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    optimizer.step()
                # Alternating schedule for optimizer steps (i.e.: GANs)
                def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                                   second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    # update generator opt every 2 steps
                    if optimizer_idx == 0:
                        if batch_idx % 2 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()
                    # update discriminator opt every 4 steps
                    if optimizer_idx == 1:
                        if batch_idx % 4 == 0 :
                            optimizer.step()
                            optimizer.zero_grad()
                    # ...
                    # add as many optimizers as you want
            Here's another example showing how to use this for more advanced things such as
            learning rate warm-up:
            .. code-block:: python
                # learning rate warm-up
                def optimizer_step(self, current_epoch, batch_idx, optimizer,
                                    optimizer_idx, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
                    # warm up lr
                    if self.trainer.global_step < 500:
                        lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr_scale * self.learning_rate
                    # update params
                    optimizer.step()
                    optimizer.zero_grad()
        Note:
            If you also override the :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_before_zero_grad`
            model hook don't forget to add the call to it before ``optimizer.zero_grad()`` yourself.
        """
        cur_lr = None
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']

        if self.args.enable_gns or self.args.enable_adascale:
            if self.args.enable_gns:
                self.gns = optimizer.gns(scale_one_batch_size=self.trainer.scale_one_bs)

            if self.args.enable_adascale:
                self.gain = optimizer.gain()
                prev_step = math.floor(self.trainer.adascale_accu_step)
                self.trainer.adascale_accu_step += self.gain
                new_step = math.floor(self.trainer.adascale_accu_step)
                scale_invariant_steps = new_step - prev_step
                # progress base scale iteration `i` by scale_invariant_steps
                self.trainer.adascale_step += scale_invariant_steps

        if on_tpu:
            xm.optimizer_step(optimizer)
        elif using_native_amp:
            self.trainer.scaler.step(optimizer)
        elif using_lbfgs:
            optimizer.step(second_order_closure)
        else:
            optimizer.step()
        tensorboard_step = self.trainer.global_step
        if self.args.enable_gns or self.args.enable_adascale:
            tensorboard_step = self.trainer.adascale_step
            if self.args.enable_gns:
                self.trainer.writer.add_scalar('Train/GNS', self.gns, tensorboard_step)

            if self.args.enable_adascale:
                self.trainer.writer.add_scalar('Train/Real Iterations', self.trainer.global_step, tensorboard_step)
                self.trainer.writer.add_scalar('Train/Gain', self.gain, tensorboard_step)
                self.trainer.writer.add_scalar('Train/Effective LR', cur_lr * self.gain, tensorboard_step)
                self.trainer.writer.add_scalar('Train/var', optimizer.nonsmooth_var[0], tensorboard_step)
                self.trainer.writer.add_scalar('Train/sqr', optimizer.nonsmooth_sqr[0], tensorboard_step)
                self.trainer.writer.add_scalar('Train/var_smooth', optimizer.var, tensorboard_step)
                self.trainer.writer.add_scalar('Train/sqr_smooth', optimizer.sqr, tensorboard_step)
                # only logging the first param group
                self.trainer.writer.add_scalar('Train/allreduced_grad_sqr', optimizer.total_grad_sqr[0],
                                  tensorboard_step)
                self.trainer.writer.add_scalar('Train/local_grad_sqr', optimizer.local_grad_sqr[0],
                                  tensorboard_step)
        self.trainer.writer.add_scalar('Train/Loss', self.cur_loss, tensorboard_step)
        self.trainer.writer.add_scalar('Train/Learning Rate', cur_lr, tensorboard_step)
        self.trainer.writer.add_scalar('Train/Scale', self.args.lr_scale, tensorboard_step)

    def save_mask(self, pred):
        if self.test_idx == 0:
            data_path = get_path(self.args)
            self.test_imgs, _ = get_test_fnames(self.args, data_path)
        fname = os.path.basename(self.test_imgs[self.test_idx]).replace("_x", "")
        np.save(os.path.join(self.save_dir, fname), pred, allow_pickle=False)
        self.test_idx += 1

    def get_train_data(self, batch):
        img, lbl = batch["image"], batch["label"]
        if self.args.dim == 2 and self.args.data2d_dim == 3:
            img, lbl = layout_2d(img, lbl)
        return img, lbl
