import torchmetrics
import torch
import torch.nn as nn

from benchmark.models.probers import ProberForBertUtterCLS

class EMOProber(ProberForBertUtterCLS):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def get_loss(self):
        return nn.MSELoss()
    
    def init_metrics(self):
        self.all_metrics = set()
        for split in ['train', 'valid', 'test']:
            # r2 score
            setattr(self, f"{split}_r2", torchmetrics.R2Score(num_outputs=2, multioutput='uniform_average'))
            self.all_metrics.add('r2')
            setattr(self, f"{split}_arousal_r2", torchmetrics.R2Score(num_outputs=1))
            self.all_metrics.add('arousal_r2')
            setattr(self, f"{split}_valence_r2", torchmetrics.R2Score(num_outputs=1))
            self.all_metrics.add('valence_r2')
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        y_pred = y_pred.detach()
        getattr(self, f"{split}_r2").update(y_pred, y)
        getattr(self, f"{split}_arousal_r2").update(y_pred[:, 0], y[:, 0])
        getattr(self, f"{split}_valence_r2").update(y_pred[:, 1], y[:, 1])
    
    @torch.no_grad()
    def log_metrics(self, split):
        self.log(f"{split}_r2", getattr(self, f"{split}_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_r2").reset()
        self.log(f"{split}_arousal_r2", getattr(self, f"{split}_arousal_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_arousal_r2").reset()
        self.log(f"{split}_valence_r2", getattr(self, f"{split}_valence_r2").compute(), sync_dist=True)
        getattr(self, f"{split}_valence_r2").reset()
    