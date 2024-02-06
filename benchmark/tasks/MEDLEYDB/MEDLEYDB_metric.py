import numpy as np
import torch
import torchmetrics
from mir_eval.melody import evaluate

class MEDLEYDBMeasure(torchmetrics.Metric):
    def __init__(self, cent_tolerance=50):
        """
        Melody extraction metric for MedleyDB.
        All default values are taken from "Computationally Efficient Dilated 
        Convolutional Model for Melody Extraction"
        Arguments:
            voicing_threshold: threshold value for estimated voicing.
            cent_tolerance: maximum absolute deviation in cents for a frequency 
                            value to be considered correct
            ref_voicing_threshold: threshold value for reference voicing.
        """
        super().__init__()
        self.cent_tolerance = cent_tolerance
        self.add_state("overall_accuracy", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        for pred, target in zip(preds, targets):
            metric = evaluate(
                target[:, 0].cpu().numpy(),
                target[:, 1].cpu().numpy(),
                pred[:, 0].cpu().numpy(),
                pred[:, 1].cpu().numpy(),
            )
            self.overall_accuracy += metric['Overall Accuracy']
            self.num += 1

    def compute(self):
        return self.overall_accuracy / self.num
