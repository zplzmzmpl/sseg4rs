import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import wandb
from lightning.pytorch.loggers import WandbLogger
import torchvision
from pytorch_lightning.callbacks import Callback


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    arg("--wandb_project", type=str, default="GeoSeg", help="WandB project name")
    arg("--wandb_entity", type=str, default=None, help="WandB entity name")
    arg("--debug", action="store_true", help="Enable debug mode with fewer iterations")
    return parser.parse_args()


class FeatureMapVisualization(Callback):
    """
    Callback to visualize feature maps from model layers periodically during training
    """
    def __init__(self, frequency=5, save_to_local=True, output_dir='feature_maps', target_layers=None):
        super().__init__()
        self.frequency = frequency
        self.save_to_local = save_to_local
        self.output_dir = output_dir
        # 修改目标层，既包括父层也包括子层
        self.target_layers = target_layers or [
            'decoder.p1', 'decoder.p1.act', 'decoder.p1.proj.1', 
            'decoder.p2', 'decoder.final_conv'
        ]
        
        # 如果需要本地保存，确保目录存在
        if self.save_to_local:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Feature maps will be saved to {os.path.abspath(self.output_dir)}")
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """在验证结束后进行特征图可视化，确保模型在评估模式"""
        if (trainer.current_epoch + 1) % self.frequency == 0:
            print(f"\n=== Visualizing feature maps at epoch {trainer.current_epoch + 1} ===")
            self._visualize_feature_maps(trainer, pl_module)
    
    def _visualize_feature_maps(self, trainer, pl_module):
        try:
            # 获取验证批次的数据
            val_dataloader = pl_module.val_dataloader()
            batch = next(iter(val_dataloader))
            img = batch['img'].to(pl_module.device)
            
            # 存储钩子和特征图
            feature_maps_dict = {}
            hook_handles = []
            
            # 记录已注册钩子的层，避免重复
            registered_modules = set()
            
            # 为目标层注册钩子函数
            for layer_name in self.target_layers:
                exact_match = False
                
                # 首先尝试精确匹配完整层名称
                for name, module in pl_module.net.named_modules():
                    if name == layer_name and name not in registered_modules:
                        def hook_fn(module, input, output, layer_name=name):
                            feature_maps_dict[layer_name] = output.detach().cpu()
                            print(f"  Captured feature maps from {layer_name} with shape {output.shape}")
                        
                        hook_handle = module.register_forward_hook(hook_fn)
                        hook_handles.append(hook_handle)
                        registered_modules.add(name)
                        exact_match = True
                        break
                
                # 如果没有精确匹配，则尝试部分匹配
                if not exact_match:
                    for name, module in pl_module.net.named_modules():
                        # 如果是子层，确保它是叶子模块
                        if layer_name in name and len(list(module.children())) == 0 and name not in registered_modules:
                            def hook_fn(module, input, output, layer_name=name):
                                feature_maps_dict[layer_name] = output.detach().cpu()
                                print(f"  Captured feature maps from {layer_name} with shape {output.shape}")
                            
                            hook_handle = module.register_forward_hook(hook_fn)
                            hook_handles.append(hook_handle)
                            registered_modules.add(name)
            
            if not registered_modules:
                print("WARNING: Could not find any of the target layers for feature map visualization!")
                print(f"Target layers: {self.target_layers}")
                print("Available layers:")
                for name, module in pl_module.net.named_modules():
                    if len(list(module.children())) == 0:  # 只打印叶子模块
                        print(f"  - {name}")
                return
            
            print(f"  Registered hooks for {len(registered_modules)} layers: {list(registered_modules)}")
            
            # 进行前向传播，触发钩子
            with torch.no_grad():
                _ = pl_module.net(img)
            
            # 移除所有钩子
            for handle in hook_handles:
                handle.remove()
            
            # 处理每个捕获的特征图
            for layer_name, feature_map in feature_maps_dict.items():
                try:
                    # 获取第一个样本的特征图
                    feature_map = feature_map[0]  # 取batch中的第一个样本
                    
                    # 如果特征图不是4D张量(通道，高，宽)，则跳过
                    if len(feature_map.shape) != 3:
                        print(f"  Skipping {layer_name}: unexpected shape {feature_map.shape}")
                        continue
                    
                    # 随机采样最多16个通道
                    num_channels = min(16, feature_map.size(0))
                    if feature_map.size(0) > num_channels:
                        indices = torch.randperm(feature_map.size(0))[:num_channels]
                        feature_map_sampled = feature_map[indices]
                    else:
                        feature_map_sampled = feature_map
                    
                    # 规范化每个通道以便更好的可视化
                    for i in range(feature_map_sampled.size(0)):
                        min_val = feature_map_sampled[i].min()
                        max_val = feature_map_sampled[i].max()
                        if max_val > min_val:
                            feature_map_sampled[i] = (feature_map_sampled[i] - min_val) / (max_val - min_val)
                    
                    # 创建特征图网格 - 将特征图转为4D (通道，1，高，宽)
                    grid = torchvision.utils.make_grid(
                        feature_map_sampled.unsqueeze(1), 
                        nrow=4, 
                        normalize=False,
                        padding=2
                    )
                    
                    # 转换为NumPy数组用于保存/显示
                    grid_np = grid.numpy().transpose((1, 2, 0))
                    
                    # 拉伸到0-255范围
                    grid_np = (grid_np * 255).astype(np.uint8)
                    
                    # 如果是单通道，转换为RGB
                    if grid_np.shape[2] == 1:
                        grid_np = np.repeat(grid_np, 3, axis=2)
                    
                    # 保存到本地
                    if self.save_to_local:
                        # 创建符合文件命名的层名
                        layer_short_name = layer_name.replace('.', '_')
                        save_path = os.path.join(
                            self.output_dir, 
                            f"epoch_{trainer.current_epoch + 1}_{layer_short_name}.png"
                        )
                        
                        # 确保目录存在
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        
                        # 保存图像
                        import cv2
                        cv2.imwrite(save_path, cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR))
                        print(f"  Saved feature map to {save_path}")
                    
                    # 记录到wandb
                    if hasattr(trainer, 'logger') and trainer.logger:
                        if isinstance(trainer.logger, list):
                            for logger in trainer.logger:
                                if isinstance(logger, WandbLogger) and hasattr(logger.experiment, "log"):
                                    logger.experiment.log({
                                        f"feature_maps/{layer_name}": wandb.Image(
                                            grid_np, 
                                            caption=f"{layer_name} - Epoch {trainer.current_epoch + 1}"
                                        )
                                    })
                        elif isinstance(trainer.logger, WandbLogger) and hasattr(trainer.logger.experiment, "log"):
                            trainer.logger.experiment.log({
                                f"feature_maps/{layer_name}": wandb.Image(
                                    grid_np, 
                                    caption=f"{layer_name} - Epoch {trainer.current_epoch + 1}"
                                )
                            })
                            print(f"  Logged feature map to wandb as feature_maps/{layer_name}")
                
                except Exception as e:
                    print(f"  Error processing feature map for {layer_name}: {e}")
            
            print(f"=== Feature map visualization completed ===\n")
            
        except Exception as e:
            import traceback
            print(f"ERROR in feature map visualization: {e}")
            print(traceback.format_exc())


class GradientMonitorCallback(Callback):
    """
    Callback to monitor and log gradient norms during training
    """
    def __init__(self, log_freq=50):
        super().__init__()
        self.log_freq = log_freq
        
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_freq == 0:
            grad_norm_dict = {}
            
            # Calculate norms for each parameter group
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    
                    # Group parameters by module name to avoid too many metrics
                    module_name = name.split('.')[0] if '.' in name else 'other'
                    if module_name not in grad_norm_dict:
                        grad_norm_dict[module_name] = []
                    
                    grad_norm_dict[module_name].append(grad_norm)
            
            # Log average gradient norm per module
            for module_name, norms in grad_norm_dict.items():
                avg_norm = sum(norms) / len(norms)
                pl_module.log(f"grad/{module_name}_norm", avg_norm, on_step=True, on_epoch=False)
            
            # Log global gradient norm
            total_norm = 0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            pl_module.log("grad/total_norm", total_norm, on_step=True, on_epoch=False)


class EarlyStopping(Callback):
    """Custom early stopping callback with patience"""
    def __init__(self, monitor='val_mIoU', patience=10, mode='max', min_delta=0.001):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.wait_count = 0
        self.best_score = None
        
    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return
            
        if isinstance(current, torch.Tensor):
            current = current.item()
            
        if self.best_score is None:
            self.best_score = current
            return
            
        if self.mode == 'min':
            improvement = (self.best_score - current) > self.min_delta
        else:
            improvement = (current - self.best_score) > self.min_delta
            
        if improvement:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            # Log directly to the logger instead of using self.log()
            if hasattr(trainer, 'logger') and trainer.logger is not None:
                if isinstance(trainer.logger, list):
                    for logger in trainer.logger:
                        if hasattr(logger, "experiment") and hasattr(logger.experiment, "log"):
                            logger.experiment.log({'early_stopping/wait_count': self.wait_count})
                else:
                    if hasattr(trainer.logger, "experiment") and hasattr(trainer.logger.experiment, "log"):
                        trainer.logger.experiment.log({'early_stopping/wait_count': self.wait_count})
            
            if self.wait_count >= self.patience:
                trainer.should_stop = True
                print(f"Early stopping triggered after {self.wait_count} epochs without improvement")


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        
        # Track loss values
        self.train_loss_values = []
        self.val_loss_values = []
        
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        hparams = {
            "num_classes": config.num_classes,
            "classes": config.classes,
            "max_epoch": config.max_epoch
        }
        if hasattr(config, "lr"):
            hparams["lr"] = config.lr
        if hasattr(config, "train_batch_size"):
            hparams["train_batch_size"] = config.train_batch_size
        if hasattr(config, "weights_name"):
            hparams["weights_name"] = config.weights_name
        if hasattr(config, "log_name"):
            hparams["log_name"] = config.log_name
        
        self.save_hyperparameters(hparams)
        
        # Automatic mixed precision if available
        self.use_amp = hasattr(config, "use_amp") and config.use_amp

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)
        
        # Log training loss
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_loss_values.append(loss.item())

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss": loss}

    def on_train_epoch_end(self):
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())
        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        
        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        if self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            class_iou_dict = {f"train_IoU_{class_name}": iou for class_name, iou in zip(self.config.classes, iou_per_class)}
            self.logger.experiment.log(class_iou_dict)
            
            # Log average training loss for this epoch
            if self.train_loss_values:
                avg_loss = sum(self.train_loss_values) / len(self.train_loss_values)
                self.logger.experiment.log({"train/epoch_loss": avg_loss})
                self.train_loss_values = []  # Reset for next epoch
        
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        loss_val = self.loss(prediction, mask)
        
        # Log validation loss
        self.log('val/loss', loss_val, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_values.append(loss_val.item())
        
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU, 'F1': F1, 'OA': OA}
        print('val:', eval_value)
        
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        if self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            class_iou_dict = {f"val_IoU_{class_name}": iou for class_name, iou in zip(self.config.classes, iou_per_class)}
            self.logger.experiment.log(class_iou_dict)
            
            # Log average validation loss for this epoch
            if self.val_loss_values:
                avg_loss = sum(self.val_loss_values) / len(self.val_loss_values)
                self.logger.experiment.log({"val/epoch_loss": avg_loss})
                self.val_loss_values = []  # Reset for next epoch
        
        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
        
        # Calculate and log learning rate
        if self.trainer.optimizers:
            for i, optimizer in enumerate(self.trainer.optimizers):
                for j, param_group in enumerate(optimizer.param_groups):
                    lr = param_group.get('lr', 0)
                    self.log(f'lr/group_{j}', lr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader
    
def print_model_structure(model):
    print("\n=== Model Structure ===")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module.__class__.__name__} (Parameters: {params:,})")
    print("=== End of Model Structure ===\n")

# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)

    # Setup callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k, 
        monitor=config.monitor,
        save_last=config.save_last, 
        mode=config.monitor_mode,
        dirpath=config.weights_path,
        filename=config.weights_name
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Feature map visualization
    feature_viz = FeatureMapVisualization(
        frequency=10,  # 增加频率以便于调试
        save_to_local=True,
        output_dir='feature_maps',
        target_layers=[
        'decoder.p1.act',           # p1的激活函数输出
        'decoder.p1.proj.1',        # 投影层输出
        'decoder.p1.shortcut.0',    # shortcut连接
        'decoder.p2.act',           # p2的激活函数输出
        'decoder.final_conv'        # 最终卷积层
        ]
    )
    callbacks.append(feature_viz)
    
    # Gradient monitoring
    grad_monitor = GradientMonitorCallback(log_freq=20)
    callbacks.append(grad_monitor)
    
    # Early stopping (if configured)
    if hasattr(config, 'early_stopping') and config.early_stopping:
        patience = config.early_stopping_patience if hasattr(config, 'early_stopping_patience') else 10
        early_stopping = EarlyStopping(
            monitor=config.monitor, 
            patience=patience,
            mode=config.monitor_mode
        )
        callbacks.append(early_stopping)
    
    # Setup loggers
    loggers = [CSVLogger('lightning_logs', name=config.log_name)]
    
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=config.log_name,
            log_model=True
        )
        
        # Log all relevant configuration
        wandb_logger.experiment.config.update({
            "model_type": config.net.__class__.__name__,
            "num_classes": config.num_classes,
            "classes": config.classes,
            "max_epoch": config.max_epoch,
            "lr": config.lr if hasattr(config, "lr") else "not specified",
            "batch_size": config.train_batch_size if hasattr(config, "train_batch_size") else "not specified",
            "optimizer": config.optimizer.__class__.__name__,
            "loss_function": config.loss.__class__.__name__,
            "monitor": config.monitor,
            "monitor_mode": config.monitor_mode,
        })
        
        loggers.append(wandb_logger)

    # Create model
    model = Supervision_Train(config)
    # print_model_structure(model)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    # Debug mode with fewer iterations if requested
    limit_train_batches = 0.1 if args.debug else 1.0
    limit_val_batches = 0.2 if args.debug else 1.0

    # Configure trainer
    trainer = pl.Trainer(
        devices=config.gpus, 
        max_epochs=config.max_epoch, 
        accelerator='auto',
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=callbacks, 
        strategy='auto',
        log_every_n_steps=5,
        logger=loggers,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        precision=16 if hasattr(config, "use_amp") and config.use_amp else 32
    )
    
    # Start training
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    
    # Clean up WandB
    if args.use_wandb:
        wandb.finish()
# python optimized_source_train.py -c config/bigbay/ftunetformer.py --use_wandb --wandb_project "GeoSeg-PlusTrain"

if __name__ == "__main__":
   main()
