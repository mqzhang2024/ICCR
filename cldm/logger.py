import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=100, max_images=4, clamp=True, increase_log_steps=True,rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join("./lightning_logs", "image_log", split)
        all_res=[]
        for k in images:
            all_res.append(images[k])
        all_res=torch.cat(all_res,dim=0) 
        grid = torchvision.utils.make_grid(all_res, nrow=4)
        if self.rescale:
            grid = (grid + 1.0) / 2.0 
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
        grid = grid.numpy()
        grid = (grid * 255).astype(np.uint8)
        filename = "gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx =  pl_module.generator.current_epoch  #pl_module. global_step
       
        if (self.check_frequency(check_idx) and 
                hasattr(pl_module.generator, "log_images") and
                callable(pl_module.generator.log_images) and
                batch_idx==0 and
                self.max_images > 0): 
            is_train = pl_module.generator.training
            if is_train:
                pl_module.generator.eval()
 
            with torch.no_grad():  
                images = pl_module.generator.log_images(batch, split=split, **self.log_images_kwargs)
            
            images['control']=(images['control']+1)/2.0
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.generator.logger.save_dir, split, images,
                           pl_module.generator.global_step, pl_module.generator.current_epoch, batch_idx)

            if is_train:
                pl_module.generator.train()

    def check_frequency(self, check_idx): 
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:  
            self.log_img(pl_module, batch, batch_idx, split="train")