import os, sys
from collections import defaultdict

import numpy as np
import sklearn
import mir_eval
from scipy.stats import mode as scipy_mode
import wandb
import argparse
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.utilities import parsing as plparsing

# from benchmark.tasks.MTT.MTT_dataset import FeatureDataset as MTTFeatureDataset, AudioDataset as MTTAudioDataset
# from benchmark.tasks.GTZAN.GTZAN_dataset import FeatureDataset as GTZANFeatureDataset
# from benchmark.tasks.GS.GS_dataset import FeatureDataset as GSFeatureDataset, AudioDataset as GSAudioDataset
# from benchmark.tasks.EMO.EMO_dataset import FeatureDataset as EMOFeatureDataset
# from benchmark.tasks.MAESTRO.MAESTRO_dataset import MaestroDataset, Sampler, TestSampler
import benchmark as bench
from benchmark.utils.get_dataloader import get_dataloaders
from benchmark.utils.get_logger import get_logger
from benchmark.utils.get_callbacks import get_callbacks 


class FinetunerForBertCLS(bench.ProberForBertUtterCLS):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.model = bench.HuBERTFeature(self.cfg.pre_trained_folder, 
                                    self.cfg.target_sr,
                                    force_half=self.cfg.force_half,
                                    disable_backprop=False)
        
        if self.cfg.finetune_freeze_CNN > 0:
            print('Freeze CNN')
            for name, param in self.model.named_parameters():
                if name.startswith("model.model.feature_extractor.conv_layers"):
                    param.requires_grad = False
                #if name.startswith("model.model.feature_projection"):
                #    param.requires_grad = False
                #if name.startswith("model.model.encoder.pos_conv_embed"):
                #    param.requires_grad = False
        if self.cfg.finetune_freeze_bottom_transformer > 0:
            #n_layers_freeze = int(args.n_tranformer_layer / 2)
            n_layers_freeze = self.cfg.finetune_freeze_bottom_transformer
            print(f'Freeze first {n_layers_freeze} layers of Transformer')
            for name, param in self.model.named_parameters():
                for i in range(n_layers_freeze):
                    if name.startswith(f"model.model.encoder.layers.{i}."):
                        param.requires_grad = False
        
    def forward(self, x):

        # the x is the audio wav preprocessed in the dataset class
        # if layer==None, self.model output [layer=13, batch, 768]
        # else output [batch, 768]
        # print(x.shape)
        if len(x.shape) == 3:
            # [batch, 1 , n_sample] -> [batch, n_sample]
            x = x.squeeze(1)
        # print(x.shape)
        if self.cfg.layer == "all":
            x = self.model(input_values=x, layer=None, reduction=self.cfg.reduction)[1:,...] # [layer=12, batch, 768]
            if len(x.shape) == 3:
                x = x.transpose(0,1) # [batch, layer=12, 768]
            # print(x.shape)
            x = self.dropout(x)
            if isinstance(self.aggregator, nn.Conv1d):
                x = self.aggregator(x).squeeze()
            else:
                weights = F.softmax(self.aggregator, dim=1)
                x = (x * weights).sum(dim=1)
            # print(x.shape)
            x = self.dropout(x)
        else:
            x = self.model(input_values=x, layer=int(self.cfg.layer), reduction=self.cfg.reduction) # => [batch, 768]

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output(x)
        # print(x.shape)
        return x

def get_model(args):
    if args.dataset in ["MTT", "MAESTRO"]:
        args.task_type = "multilabel"
    elif args.dataset in ["GTZAN", "GS", "NSynthI"]:
        args.task_type = "multiclass"
    elif args.dataset in ["EMO"]:
        args.task_type = "regression"
    else:
        raise NotImplementedError(f"get_prober() of dataset {args.dataset} not implemented")

    if args.dataset in ["MTT", "GTZAN", "GS", "EMO", "NSynthI"]:
        return FinetunerForBertCLS(args)
    else:
        raise NotImplementedError(f"get_prober() of dataset {args.dataset} not implemented")

def main(args):
    pl.seed_everything(1234)
    logger = get_logger(args, train_type='finetune')
    model = get_model(args)
    train_loader, valid_loader, test_loader = get_dataloaders(args, dataset_type='audio')
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, 
                                            logger=logger, 
                                            callbacks=callbacks, 
                                            default_root_dir=f"./data/{args.dataset}", 
                                            fast_dev_run=args.debug)
    if args.eval_only:
        assert args.eval_ckpt_path
        trainer.validate(dataloaders=valid_loader, ckpt_path=args.eval_ckpt_path)
        trainer.test(dataloaders=test_loader, ckpt_path=args.eval_ckpt_path)
        """ What is this?"""
        # method 1: manually loading and assign + testing
        # requires namespace converter, not recommended, use this implementation when requiring model hparams modification
        # model = model.load_from_checkpoint(checkpoint_path=args.eval_ckpt_path, map_location=None,) 
        # trainer.test(model=model, dataloaders=test_loader)
        # # method 2: directly testing
        # trainer.validate(model=model, dataloaders=valid_loader, ckpt_path=args.eval_ckpt_path)
        # # print(best_strategy_name)
        # trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.eval_ckpt_path)
        # print(logger.experiment.config)

    else:
        trainer.fit(model, train_loader, valid_loader)
        # TODO: do we need to valid again? it's already done in trainer.fit()
        trainer.validate(dataloaders=valid_loader, ckpt_path='best')
        if not args.debug:
            trainer.test(dataloaders=test_loader, ckpt_path='best')
            if args.save_best_to is not None:
                # does it really save the best model?
                trainer.save_checkpoint(args.save_best_to)
