import torch
from torch import nn
import torchmetrics

import benchmark as bench

class MAESTROProber(bench.ProberForBertSeqLabel):
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

    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            setattr(self, f"{split}_prec", torchmetrics.Precision(task='binary', threshold=self.cfg.frame_threshold))
            self.all_metrics.add('prec')
            setattr(self, f"{split}_recall", torchmetrics.Recall(task='binary', threshold=self.cfg.frame_threshold))
            self.all_metrics.add('recall')
            setattr(self, f"{split}_f1", torchmetrics.F1Score(task='binary', threshold=self.cfg.frame_threshold))
            self.all_metrics.add('f1')
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        """update metrics during train/valid/test step"""
        y = y.int() # [B, T, C], C is the number of classes
        y_pred = y_pred.detach()
        y_pred = torch.sigmoid(y_pred)

        # y = torch.flatten(y, end_dim=1) # [BxT, C]
        # y_pred = torch.flatten(y_pred, end_dim=1) # [BxT, C]

        y_flat = torch.flatten(y) # [BxTxC]
        y_pred_flat = torch.flatten(y_pred) # [BxTxC]

        getattr(self, f"{split}_prec").update(y_pred_flat, y_flat)
        getattr(self, f"{split}_recall").update(y_pred_flat, y_flat)
        getattr(self, f"{split}_f1").update(y_pred_flat, y_flat)
    
    @torch.no_grad()
    def log_metrics(self, split):
        self.log(f"{split}_prec", getattr(self, f"{split}_prec").compute(), sync_dist=True)
        getattr(self, f"{split}_prec").reset()
        self.log(f"{split}_recall", getattr(self, f"{split}_recall").compute(), sync_dist=True)
        getattr(self, f"{split}_recall").reset()
        self.log(f"{split}_f1", getattr(self, f"{split}_f1").compute(), sync_dist=True)
        getattr(self, f"{split}_f1").reset()
    