import torch
from torchmetrics import Metric


class GlobalSDR(Metric):
    """ Global SDR """
    def __init__(self):
        super().__init__()
        self.add_state("sdr", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0), dist_reduce_fx="sum")
                       
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert target.shape == preds.shape,\
               (f"Shape mismatch between references ({target.shape}) and "
                f"estimates ({preds.shape}).")
        
        delta = 1e-7  # avoid numerical errors
        num = torch.sum(torch.square(target), axis=(1, 2)).squeeze()
        den = torch.sum(torch.square(target - preds), axis=(1, 2)).squeeze()
        num += delta
        den += delta

        self.sdr += 10 * torch.log10(num / den)
        self.num += 1
    
    def compute(self):
        return self.sdr / self.num