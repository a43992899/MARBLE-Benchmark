import numpy as np
import torch
import torchmetrics
import mir_eval
from mir_eval.beat import validate
from madmom.features.beats import DBNBeatTrackingProcessor


class BCEBeatFMeasure(torchmetrics.Metric):
    def __init__(self, label_freq=75, downbeat=False, metric_type="both"):
        super().__init__()
        self.label_freq = label_freq
        self.downbeat = downbeat
        self.type = metric_type
        self.add_state("f_measure", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("matching", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("estimate", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("reference", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("f_measure_db", default=torch.tensor(0.0), dist_reduce_fx="mean")
        self.add_state("matching_db", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("estimate_db", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("reference_db", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target are assumed to be sequences of beat times
        def get_idx(x):
            indices = np.nonzero(x == 1)  #shape (2251,)
            indices = indices[0].flatten()
            indices = indices.astype(np.float32) / self.label_freq
            return indices
        
        proc = DBNBeatTrackingProcessor(fps=self.label_freq)
        estimated_beats = proc(preds[0].cpu().numpy())
        
        reference_beats = get_idx(target[:,0].cpu().numpy())
        f_measure_threshold=0.07
        mir_eval.beat.validate(reference_beats, estimated_beats)
        # # When estimated beats are empty, no beats are correct; metric is 0
        # if estimated_beats.size == 0 or reference_beats.size == 0:
        #     return 0.
        # Compute the best-case matching between reference and estimated locations
        matching = mir_eval.util.match_events(reference_beats,
                                            estimated_beats,
                                            f_measure_threshold)
        self.matching += len(matching)
        self.estimate += len(estimated_beats)
        self.reference += len(reference_beats)
        
        # proc = DBNDownBeatTrackingProcessor()
        proc = DBNBeatTrackingProcessor(min_bpm=18, max_bpm=72, fps=self.label_freq)
        estimated_beats = proc(preds[1].cpu().numpy())
        reference_beats = get_idx(target[:,1].cpu().numpy())
        mir_eval.beat.validate(reference_beats, estimated_beats)
        matching = mir_eval.util.match_events(reference_beats,
                                            estimated_beats,
                                            f_measure_threshold)
        self.matching_db += len(matching)
        self.estimate_db += len(estimated_beats)
        self.reference_db += len(reference_beats)
        
    def compute(self):
        def calculate(matching, estimate, reference):
            if estimate == 0 or reference == 0:
                return torch.tensor(0.0)
            precision = float(matching)/estimate
            recall = float(matching)/reference
            if precision == 0 and recall == 0:
                f_measure = 0.0
            else:
                f_measure = mir_eval.util.f_measure(precision, recall)
            return torch.tensor(f_measure)
        if self.type == "beat":
            return calculate(self.matching, self.estimate, self.reference)
        elif self.type == "downbeat":
            return calculate(self.matching_db, self.estimate_db, self.reference_db)
        else:
            return (calculate(self.matching, self.estimate, self.reference) + calculate(self.matching_db, self.estimate_db, self.reference_db)) / 2
        

class BeatFMeasure(BCEBeatFMeasure):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds and target are assumed to be sequences of beat times
        def get_idx(x):
            x = x.squeeze(0)
            if self.downbeat:
                indices = torch.nonzero((x == 2), as_tuple=False)
            else:
                indices = torch.nonzero((x == 1) | (x == 2), as_tuple=False)
            indices = torch.flatten(indices)
            indices = indices.float() / self.label_freq
            return indices.cpu().numpy()
        
        batch_size = preds.shape[0]
        for batch in range(batch_size):
            proc = DBNBeatTrackingProcessor(fps=self.label_freq)
            estimated_beats = proc(preds[batch][1].cpu().numpy())
            
            reference_beats = get_idx(target[batch])
            f_measure_threshold=0.07
            mir_eval.beat.validate(reference_beats, estimated_beats)
            matching = mir_eval.util.match_events(reference_beats,
                                                estimated_beats,
                                                f_measure_threshold)
            self.matching += len(matching)
            self.estimate += len(estimated_beats)
            self.reference += len(reference_beats)
        
    def compute(self):
        if self.estimate == 0 or self.reference == 0:
            return torch.tensor(0.0)
        precision = float(self.matching)/self.estimate
        recall = float(self.matching)/self.reference
        if precision == 0 and recall == 0:
            f_measure = 0.0
        else:
            f_measure = mir_eval.util.f_measure(precision, recall)
        
        return torch.tensor(f_measure)
    