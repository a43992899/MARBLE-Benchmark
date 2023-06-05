import torch
from torch import nn
import torchmetrics

import benchmark as bench
from benchmark.tasks.GTZAN.GTZANBT_metric import BCEBeatFMeasure, BeatFMeasure

class GTZANBTProber(bench.ProberForBertSeqLabel):
    """MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Sequence labeling tasks are token level classification tasks, in MIR, it is usually used for
    onset detection, note transcription, etc.

    This class supports learnable weighted sum over different layers of features.

    TODO: support official evaluation strategy
    TODO: enable biLSTM
    TODO: fix refresh rate
    TODO: support define torchmetrics
    TODO: support MAESTRO fast dataset
    """
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_loss(self):
        if self.hparams.loss_weight:
            self.loss_weight = torch.Tensor(eval(self.hparams.loss_weight))
        else:
            self.loss_weight = None
        
        if self.hparams.task_type == "multilabel":
            return nn.BCEWithLogitsLoss(weight=self.loss_weight)
        elif self.hparams.task_type == "multiclass":
            return nn.CrossEntropyLoss(weight=self.loss_weight)
        else:
            raise NotImplementedError(f"get_loss() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")

    def init_metrics(self):
        self.all_metrics = set()
        for split in ['train', 'valid', 'test']:
            if self.hparams.task_type == 'multilabel':
                setattr(self, f"{split}_beat_f", BCEBeatFMeasure(label_freq=self.hparams.token_rate, metric_type="beat"))
                self.all_metrics.add("beat_f")
                setattr(self, f"{split}_downbeat_f", BCEBeatFMeasure(label_freq=self.hparams.token_rate, metric_type="downbeat"))
                self.all_metrics.add("downbeat_f")
                setattr(self, f"{split}_meanbeat_f", BCEBeatFMeasure(label_freq=self.hparams.token_rate, metric_type="mean"))
                self.all_metrics.add("meanbeat_f")
            elif self.hparams.task_type == 'multiclass':
                setattr(self, f"{split}_beat_f", BeatFMeasure(label_freq=self.hparams.token_rate))
                self.all_metrics.add("beat_f")
            else:
                raise NotImplementedError(f"{self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        """update metrics during train/valid/test step"""
        y = y.int() # [B, T, C], C is the number of classes
        if self.hparams.task_type == 'multilabel':
            y_pred = y_pred.detach()
            y_pred = torch.sigmoid(y_pred)
            y = torch.flatten(y, end_dim=1) # [BxT, C]
            y_pred = torch.flatten(y_pred, end_dim=1) # [BxT, C]
        else:
            y_pred = y_pred.detach()
            y_pred = torch.softmax(y_pred, dim=1)
        
        if self.hparams.task_type == 'multilabel':
            getattr(self, f"{split}_beat_f").update(y_pred, y)
            getattr(self, f"{split}_downbeat_f").update(y_pred, y)
            getattr(self, f"{split}_meanbeat_f").update(y_pred, y)
        else:
            getattr(self, f"{split}_beat_f").update(y_pred, y)
    
    @torch.no_grad()
    def log_metrics(self, split):
        if self.hparams.task_type == 'multilabel':
            self.log(f"{split}_beat_f", getattr(self, f"{split}_beat_f").compute(), sync_dist=True)
            getattr(self, f"{split}_beat_f").reset()
            self.log(f"{split}_downbeat_f", getattr(self, f"{split}_downbeat_f").compute(), sync_dist=True)
            getattr(self, f"{split}_downbeat_f").reset()
            self.log(f"{split}_meanbeat_f", getattr(self, f"{split}_meanbeat_f").compute(), sync_dist=True)
            getattr(self, f"{split}_meanbeat_f").reset()
        else:
            self.log(f"{split}_beat_f", getattr(self, f"{split}_beat_f").compute(), sync_dist=True)
            getattr(self, f"{split}_beat_f").reset()
    