from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.set_dataset import MyDataset
from muti_control.ICCR import ICCR
from cldm.logger import ImageLogger


# Configs
batch_size = 22
logger_freq = 1
json_root='split_data'          # root directory of the JSON data file 
use_FM=True
use_PiP=True
pip_face_file='meanface.txt'    # meanface file from PiPNet
pip_weight_file='pipnet.pth'    # weight file from PiPNet
sd_file='./models/control_sd15_ini.ckpt'

# model
model = ICCR(batch_size,use_FM,use_PiP,sd_file,pip_face_file,pip_weight_file)

# dataset
train_dataset = MyDataset(json_root,'train.json')
train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

val_dataset = MyDataset(json_root,'valid.json')
val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=True)

# log
logger = ImageLogger(batch_frequency=logger_freq)

# trainer
trainer = pl.Trainer(devices=[0],max_epochs=300, precision=32,callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=10, save_top_k=-1,save_last=True),pl.callbacks.ModelCheckpoint(save_top_k=1,monitor='G_loss',mode='min',filename='best_{epoch:02d}') ,logger])   
trainer.fit(model, train_dataloader,val_dataloader)   