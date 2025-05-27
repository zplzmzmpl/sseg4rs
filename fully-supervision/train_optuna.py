import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg
import os
import torch
from torch import nn
# import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
import wandb
from lightning.pytorch.loggers import WandbLogger
import optuna
# from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import Callback
import copy
# import shutil

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class CustomPruningCallback(Callback):
    
    def __init__(self, trial, monitor="val_loss", mode="minimize"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.mode = mode
        
    def on_validation_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        metrics = trainer.callback_metrics
        current_score = metrics.get(self.monitor)
        if current_score is None:
            return
        
        current_score = float(current_score)
        
        if self.mode == "minimize":
            current_score = -current_score
            
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


class CleanCheckpointCallback(Callback):
    """
    Callback to remove checkpoint files after trial completion to save storage space
    """
    def __init__(self, checkpoint_dir, trial_number):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.trial_number = trial_number
    
    def on_fit_end(self, trainer, pl_module):
        # Keep track of only the best trial checkpoint
        if hasattr(trainer, 'callback_metrics') and trainer.callback_metrics.get('val_F1', 0) < 0.5:
            # Delete checkpoints for this trial if performance is below threshold
            checkpoint_pattern = os.path.join(self.checkpoint_dir, f"optuna_trial_{self.trial_number}*")
            import glob
            for file_path in glob.glob(checkpoint_pattern):
                try:
                    os.remove(file_path)
                    print(f"Removed checkpoint: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    arg("--wandb_project", type=str, default="GeoSeg", help="WandB project name")
    arg("--wandb_entity", type=str, default=None, help="WandB entity name")
    # Optuna
    arg("--use_optuna", action="store_true", help="Use Optuna for hyperparameter optimization")
    arg("--optuna_study", type=str, default="geoseg_optimization", help="Optuna study name")
    arg("--optuna_trials", type=int, default=20, help="Number of Optuna trials")
    arg("--optuna_epochs", type=int, default=5, help="Epochs per trial during optimization")
    arg("--visualize_optuna", action="store_true", help="Visualize Optuna results")
    arg("--optuna_storage", type=str, default=None, help="Optuna storage URL")
    arg("--save_optuna_checkpoints", action="store_true", help="Save model checkpoints during optimization")
    return parser.parse_args()


def process_model_params(net, backbone=False):
    """
    extract backbone paras & other paras
    """
    backbone_params = []
    other_params = []
    
    for name, param in net.named_parameters():
        if "backbone" in name and backbone:
            backbone_params.append(param)
        elif "backbone" not in name and not backbone:
            other_params.append(param)
    
    return backbone_params if backbone else other_params


class OptunaTrainer:
    def __init__(self, base_config, args):
        self.base_config = base_config
        self.args = args
        # extract fixed parameters
        self.fixed_params = {
            'net': base_config.net,
            'loss': base_config.loss,
            'num_classes': base_config.num_classes,
            'classes': base_config.classes,
            'train_loader': base_config.train_loader,
            'val_loader': base_config.val_loader,
            'gpus': base_config.gpus,
            'weights_path': base_config.weights_path,
            'max_epoch': args.optuna_epochs,
            'monitor': 'val_F1',
            'monitor_mode': 'max',
            'save_top_k': 1,
            'save_last': False,
            'check_val_every_n_epoch': 1,
        }

    def objective(self, trial):
        config_dict = copy.deepcopy(self.fixed_params)
        
        config_dict['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        
        config_dict['backbone_lr'] = trial.suggest_float('backbone_lr', 1e-6, 1e-3, log=True)

        config_dict['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        config_dict['backbone_weight_decay'] = trial.suggest_float('backbone_weight_decay', 1e-6, 1e-3, log=True)
        
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
        config_dict['train_batch_size'] = batch_size
        config_dict['val_batch_size'] = batch_size
        
        config_dict['weights_name'] = f"optuna_trial_{trial.number}"
        config_dict['log_name'] = f"optuna_trial_{trial.number}"
        
        trial_config = type('Config', (object,), config_dict)
        optimizer = torch.optim.AdamW([
            {'params': process_model_params(trial_config.net, False), 'lr': trial_config.lr, 'weight_decay': trial_config.weight_decay},
            {'params': process_model_params(trial_config.net, True), 'lr': trial_config.backbone_lr, 'weight_decay': trial_config.backbone_weight_decay}
        ])
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=trial_config.max_epoch,
            T_mult=1,
            eta_min=1e-6
        )
        
        trial_config.optimizer = optimizer
        trial_config.lr_scheduler = lr_scheduler
        trial_config.use_aux_loss = False
        
        model = Supervision_Train(trial_config)
        
        early_stop_callback = EarlyStopping(
            monitor=trial_config.monitor,
            min_delta=0.00,
            patience=3,
            verbose=True,
            mode=trial_config.monitor_mode
        )
        
        pruning_callback = CustomPruningCallback(
            trial=trial,monitor=trial_config.monitor, 
            mode="maximize" if trial_config.monitor_mode == "max" else "minimize")
        
        callbacks = [early_stop_callback, pruning_callback]
        
        if self.args.save_optuna_checkpoints:
            checkpoint_callback = ModelCheckpoint(
                save_top_k=trial_config.save_top_k,
                monitor=trial_config.monitor,
                save_last=trial_config.save_last,
                mode=trial_config.monitor_mode,
                dirpath=trial_config.weights_path,
                filename=trial_config.weights_name
            )
            callbacks.append(checkpoint_callback)
        else:
            # Add clean checkpoint callback to remove checkpoints
            clean_callback = CleanCheckpointCallback(
                checkpoint_dir=trial_config.weights_path,
                trial_number=trial.number
            )
            callbacks.append(clean_callback)
        
        loggers = [CSVLogger('lightning_logs', name=trial_config.log_name)]
        
        if self.args.use_wandb:
            wandb_logger = WandbLogger(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=f"optuna_{trial_config.log_name}",
                group="optuna_trials",
                log_model=False
            )
            
            params = {
                "trial_number": trial.number,
                "lr": trial_config.lr,
                "backbone_lr": trial_config.backbone_lr,
                "weight_decay": trial_config.weight_decay,
                "backbone_weight_decay": trial_config.backbone_weight_decay,
                "batch_size": trial_config.train_batch_size,
            }
            wandb_logger.log_hyperparams(params)
            loggers.append(wandb_logger)
            
        trainer = pl.Trainer(
            devices=trial_config.gpus,
            max_epochs=trial_config.max_epoch,
            accelerator='auto',
            check_val_every_n_epoch=trial_config.check_val_every_n_epoch,
            callbacks=callbacks,
            strategy='auto',
            log_every_n_steps=5,
            logger=loggers
        )
        
        trainer.fit(model=model)
        
        return trainer.callback_metrics[trial_config.monitor].item()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)
        hparams = {
        "num_classes": config.num_classes,
        "classes": config.classes,
        "max_epoch": config.max_epoch}
        if hasattr(config, "lr"):
            hparams["lr"] = config.lr
        if hasattr(config, "train_batch_size"):
            hparams["train_batch_size"] = config.train_batch_size
        if hasattr(config, "weights_name"):
            hparams["weights_name"] = config.weights_name
        if hasattr(config, "log_name"):
            hparams["log_name"] = config.log_name
        
        self.save_hyperparameters(hparams)

    def forward(self, x):
        # only net is used in the prediction/inference
        seg_pre = self.net(x)
        return seg_pre

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']

        prediction = self.net(img)
        loss = self.loss(prediction, mask)

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
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        if self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            class_iou_dict = {f"train_IoU_{class_name}": iou for class_name, iou in zip(self.config.classes, iou_per_class)}
            self.logger.experiment.log(class_iou_dict)
        
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        if self.logger and hasattr(self.logger, "experiment") and hasattr(self.logger.experiment, "log"):
            class_iou_dict = {f"val_IoU_{class_name}": iou for class_name, iou in zip(self.config.classes, iou_per_class)}
            self.logger.experiment.log(class_iou_dict)
        
        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self.config.train_loader

    def val_dataloader(self):
        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(24)
    
    if args.use_optuna:
        print(f"Starting Optuna hyperparameter optimization with {args.optuna_trials} trials...")
        
        study = optuna.create_study(
            study_name=args.optuna_study,
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            storage=args.optuna_storage
        )
        
        optuna_trainer = OptunaTrainer(config, args)
        
        study.optimize(optuna_trainer.objective, n_trials=args.optuna_trials)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
            
        if args.visualize_optuna:
            try:
                import plotly
                from optuna.visualization import (
                    plot_optimization_history, 
                    plot_param_importances,
                    plot_contour,
                    plot_slice,
                    plot_parallel_coordinate,
                    plot_edf,
                    plot_intermediate_values
                )
                
                vis_dir = os.path.join(config.weights_path, "optuna_vis")
                os.makedirs(vis_dir, exist_ok=True)
                
                # Save optimization history
                fig1 = plot_optimization_history(study)
                fig1.write_html(os.path.join(vis_dir, "optimization_history.html"))
                
                # Save parameter importance
                fig2 = plot_param_importances(study)
                fig2.write_html(os.path.join(vis_dir, "param_importances.html"))
                
                # Save contour plot
                fig3 = plot_contour(study)
                fig3.write_html(os.path.join(vis_dir, "contour.html"))
                
                # Save slice plot
                fig4 = plot_slice(study)
                fig4.write_html(os.path.join(vis_dir, "slice.html"))
                
                # New: Save parallel coordinate plot
                fig5 = plot_parallel_coordinate(study)
                fig5.write_html(os.path.join(vis_dir, "parallel_coordinate.html"))
                
                # New: Save empirical distribution function plot
                fig6 = plot_edf(study)
                fig6.write_html(os.path.join(vis_dir, "edf.html"))
                
                # New: Save intermediate values plot
                fig7 = plot_intermediate_values(study)
                fig7.write_html(os.path.join(vis_dir, "intermediate_values.html"))
                
                print(f"Optuna visualization saved to {vis_dir}")
                
                if args.use_wandb:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=f"{config.log_name}_optuna_visualization",
                        config=trial.params
                    )

                    wandb.log({
                        "optimization_history": wandb.Html(os.path.join(vis_dir, "optimization_history.html")),
                        "param_importances": wandb.Html(os.path.join(vis_dir, "param_importances.html")),
                        "contour": wandb.Html(os.path.join(vis_dir, "contour.html")),
                        "slice": wandb.Html(os.path.join(vis_dir, "slice.html")),
                        "parallel_coordinate": wandb.Html(os.path.join(vis_dir, "parallel_coordinate.html")),
                        "edf": wandb.Html(os.path.join(vis_dir, "edf.html")),
                        "intermediate_values": wandb.Html(os.path.join(vis_dir, "intermediate_values.html"))
                    })
                    
                    wandb.finish()
            
            except ImportError as e:
                print(f"Warning can't import visualization lib: {e}")
                print("pip install plotly")
        
        # Save the best parameters to a file
        best_params_file = os.path.join(config.weights_path, f"{args.optuna_study}_best_params.txt")
        with open(best_params_file, 'w') as f:
            f.write(f"Best trial value: {trial.value}\n")
            f.write("Best parameters:\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Best parameters saved to {best_params_file}")

# python train_supervision.py -c config/bigbay/ftunetformer.py
# python train_supervision.py -c config/bigbay/ftunetformer.py --use_wandb --wandb_project "GeoSeg" --wandb_entity "dabuliu-china-university-of-geosciences"
# python train_supervision.py -c config/bigbay/ftunetformer.py --use_optuna --optuna_trials 20 --optuna_epochs 5 --visualize_optuna --use_wandb --save_optuna_checkpoints --wandb_project "GeoSeg_Opz" --wandb_entity "dabuliu-china-university-of-geosciences"

if __name__ == "__main__":
   main()