from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.bigbay_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params

wandb_api = "fb408fd7c242d2b96689adf2a9107783a9ef5472"
# training hparam
early_stopping = True  # Optional, defaults to False
early_stopping_patience = 15  # Optional, defaults to 10
use_amp = True  # Enable automatic mixed precision
max_epoch = 45
ignore_index = len(CLASSES)
train_batch_size = 4
val_batch_size = 4
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "ftunetformer-768-crop-ms-e45"
weights_path = "model_weights/bigbay/{}".format(weights_name)
test_weights_name = "ftunetformer-768-crop-ms-e45"
log_name = 'bigbay/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader

train_dataset = LandUseDataset(data_root='E:/2025/full_landsat_20cls/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug)

val_dataset = LandUseDataset(data_root='E:/2025/full_landsat_20cls/val',transform=val_aug)
test_dataset = LandUseDataset(data_root='E:/2025/full_landsat_20cls/test',transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          persistent_workers=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        persistent_workers=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)