# import os, sys
from collections import defaultdict

import numpy as np
# import sklearn
import mir_eval
from scipy.stats import mode as scipy_mode
# import wandb
import argparse
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.utilities import parsing as plparsing

from benchmark.extract_hubert_features import HuBERTFeature
from benchmark.utils.losses import bce
from benchmark.utils.get_dataloader import get_dataloaders
from benchmark.utils.get_logger import get_logger
from benchmark.utils.get_callbacks import get_callbacks 
from benchmark.GTZANRhythm.GTZAN_rhythm_metric import BeatFMeasure, BCEBeatFMeasure

class MLPProberBase(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        # in case it's recover from a checkpoint
        # TODO: change to the PL official implementation for hyper parameters saving
        if not isinstance(self.cfg, argparse.Namespace):
            self.cfg = plparsing.AttributeDict(self.cfg)
            print(self.cfg)
            
        if self.cfg.loss_weight:
            self.loss_weight = torch.Tensor(eval(self.cfg.loss_weight))
        else:
            self.loss_weight = None
        d = self.cfg.num_features
        self.hidden_layer_sizes = eval(self.cfg.hidden_layer_sizes)
        self.num_layers = len(self.hidden_layer_sizes)
        for i, ld in enumerate(self.hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld
        self.output = nn.Linear(d, self.cfg.num_outputs)
        self.dropout = nn.Dropout(p=self.cfg.dropout_p)
        self.loss = self.get_loss()
        self.init_metrics()
        self.monitor_mode = 'min' if 'loss' in self.cfg.monitor else 'max'

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = self.dropout(x)
            x = F.relu(x)
        output = self.output(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        if batch_idx % self.cfg.logging_steps == 0:
            with torch.no_grad():
                log_dict = self.compute_metrics('train', y, y_pred, loss)
                self.log_dict(log_dict, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        log_dict = self.compute_metrics('valid', y, y_pred, loss)
        self.log_dict(log_dict, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        log_dict = self.compute_metrics('test', y, y_pred, loss)
        self.log_dict(log_dict, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.l2_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.monitor_mode, factor=0.5, patience=self.cfg.lr_scheduler_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.cfg.monitor}
    
    def get_loss(self):
        raise NotImplementedError(f"get_loss() not implemented in {self.__class__.__name__}")

    def init_metrics(self):
        raise NotImplementedError(f"init_metrics() not implemented in {self.__class__.__name__}")
        
    def compute_metrics(self, split, y, y_pred, loss):
        raise NotImplementedError(f"compute_metrics() not implemented in {self.__class__.__name__}")


class ProberForBertCLS(MLPProberBase):
    """MLP Prober for utterance-level classification tasks with BERT like 12 layer features (HuBERT, Data2vec)
    This class supports learnable weighted sum over different layers of features.
    Note that we assume the features are pre-extracted and time averaged (since these audio berts do not have CLS token).
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.layer == "all":
            # use learned weights to aggregate features
            if self.cfg.normalized_weight_sum:
                self.aggregator = nn.Parameter(torch.randn((1, self.cfg.n_tranformer_layer, 1)))
            else:
                self.aggregator = nn.Conv1d(in_channels=self.cfg.n_tranformer_layer, out_channels=1, kernel_size=1)
        
        if self.cfg.test_ensemble:
            self.valid_best_ensemble_score = float("-inf")
            self.valid_best_ensemble_strategy = None
    
    def forward(self, x):
        # x.shape = (batch, n_tranformer_layer + 1, 1, 768)
        if self.cfg.layer == "all":
            x = self.dropout(x)
            if isinstance(self.aggregator, nn.Conv1d):
                x = self.aggregator(x).squeeze()
            else:
                weights = F.softmax(self.aggregator, dim=1)
                x = (x * weights).sum(dim=1)
            x = self.dropout(x)

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.output(x)
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.cfg.test_ensemble:
                x, y, meta_idx, class_in_str = batch
            else:
                x, y = batch
            y_pred = self(x)
            loss = self.loss(y_pred, y)
            log_dict = self.compute_metrics('valid', y, y_pred, loss)
            self.log_dict(log_dict, prog_bar=True)
        if self.cfg.test_ensemble:
            return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), np.array(class_in_str)
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.cfg.test_ensemble:
                x, y, meta_idx, class_in_str = batch
            else:
                x, y = batch
            y_pred = self(x)
            loss = self.loss(y_pred, y)
            
            log_dict = self.compute_metrics('test', y, y_pred, loss)
            self.log_dict(log_dict, prog_bar=True)
        if self.cfg.test_ensemble:
            return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), class_in_str
    
    def log_weights(self):
        weights = F.softmax(self.aggregator, dim=1).squeeze().detach().cpu().numpy()
        log_dict = {f"layer_{i+1}_weight": w.item() for i, w in enumerate(weights)}
        return log_dict

    def training_epoch_end(self, outputs):
        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                with torch.no_grad():
                    log_dict = self.log_weights()
                    self.log_dict(log_dict, prog_bar=False)
    
    def validation_epoch_end(self, outputs):
        if self.cfg.test_ensemble:
            with torch.no_grad():
                log_ensemble_results = self.predict_result_ensemble(outputs) # [(strategy_name, predicted_label, label), ...]
                best_strategy_name = log_ensemble_results.pop("best_ensemble_strategy")
                if log_ensemble_results["best_ensemble_score"] > self.valid_best_ensemble_score:
                    self.valid_best_ensemble_score = log_ensemble_results["best_ensemble_score"]
                    self.valid_best_ensemble_strategy = best_strategy_name
                if not self.cfg.wandb_off:
                    self.logger.experiment.log( {"valid_best_ensemble_strategy": best_strategy_name}) # equals to wandb.log()
                log_ensemble_results = dict(("valid_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
                self.log_dict(log_ensemble_results, prog_bar=False)
    
    def test_epoch_end(self, outputs):
        '''
        @yizhilll: https://github.com/Lightning-AI/lightning/discussions/9914
        the test_step outputs could be aggregated if using DataParrallel, we don't support this yet.
        outputs
        '''
        with torch.no_grad():
            if self.cfg.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict = self.log_weights()
                    self.log_dict(log_dict, prog_bar=False)
            if self.cfg.test_ensemble:
                log_ensemble_results = self.predict_result_ensemble(outputs) # [(strategy_name, predicted_label, label), ...]
                # additional handling the string log
                log_ensemble_results.pop("best_ensemble_strategy") # no need for test_best_strategy_name
                if self.valid_best_ensemble_strategy is not None:
                    log_ensemble_results["ensemble_score_val-select"] = log_ensemble_results[f"ensemble_{self.valid_best_ensemble_strategy}_score"]
                # self.logger.experiment.config["test_best_ensemble_strategy"] = log_ensemble_results.pop("best_ensemble_strategy")
                log_ensemble_results = dict(("test_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
                self.log_dict(log_ensemble_results, prog_bar=False)

    def predict_result_ensemble(self, outputs_with_meta_ids):
        '''
        adapted from jukemir
        input: 
            outputs_with_meta_ids = [(), ...] <= provided by test_step
        return:
            comparisions = [(strategy_name, predicted_label, label), ...] <= mir_eval format
            (torchmetrics.Accuracy support both predicted_label and predicted_logits input)
        '''
        # meta info for GS dataset
        # id_to_label = DATASET_TO_ATTRS["giantsteps_clips"]["labels"]
        classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        class2id = {c: i for i, c in enumerate(classes)}
        id2class = {v: k for k, v in class2id.items()}
        id_to_label = id2class

        # retrieve all the predicitions and labels
        y = []
        clip_logits = []
        y_meta_id = []
        class_info = []
        for label, pred_logit, meta_id, class_in_str in outputs_with_meta_ids:
            y.append(label)
            clip_logits.append(pred_logit)
            y_meta_id.append(meta_id)
            class_info += list(class_in_str)

        y_meta_id = np.hstack(y_meta_id) # -> (2406,)
        clip_labels = np.hstack(y) # -> (2406, )
        clip_logits =  np.vstack(clip_logits) # -> (2406, 24)
        clip_preds = np.argmax(clip_logits, axis=1)
        class_info = np.array(class_info)
        with torch.no_grad():
            clip_probs = (
                F.softmax(torch.tensor(clip_logits, device=self.device), dim=-1)
                .cpu()
                .numpy()
            )
        # merge the predictions and labels according to their meta audio id
        # largely adapted from jukemir
        song_uid_to_clip_idxs = defaultdict(list) # meta_id -> actual index in clip_labels&clip_logits
        song_uid_to_label = {}
        for clip_idx, (song_uid, label) in enumerate(zip(y_meta_id, clip_labels)):
            song_uid_to_clip_idxs[song_uid].append(clip_idx)
            if song_uid in song_uid_to_label:
                assert song_uid_to_label[song_uid] == label
            song_uid_to_label[song_uid] = label
        song_uids = sorted(song_uid_to_clip_idxs.keys())
        song_labels = np.array(
            [song_uid_to_label[song_uid] for song_uid in song_uids]
        )
        # Ensemble predictions
        ensemble_strategy_to_song_preds = defaultdict(list)
        for song_uid in song_uids:
            clip_idxs = song_uid_to_clip_idxs[song_uid]

            song_clip_logits = clip_logits[clip_idxs]
            song_clip_preds = clip_preds[clip_idxs]
            song_clip_probs = clip_probs[clip_idxs]
            ensemble_strategy_to_song_preds["vote"].append(
                scipy_mode(song_clip_preds, keepdims=True).mode[0]
            )
            ensemble_strategy_to_song_preds["max"].append(
                song_clip_logits.max(axis=0).argmax()
            )
            ensemble_strategy_to_song_preds["gmean"].append(
                song_clip_logits.mean(axis=0).argmax()
            )
            ensemble_strategy_to_song_preds["mean"].append(
                song_clip_probs.mean(axis=0).argmax()
            )

        # Compute all metrics
        comparisons = [
            (
                "clip",
                np.argmax(clip_probs, axis=1),
                clip_labels,
            )
        ]
        comparisons += [
            (f"ensemble_{strategy_name}", np.array(strategy_preds), song_labels)
            for strategy_name, strategy_preds in ensemble_strategy_to_song_preds.items()
        ]

        # return comparisons
        def _compute_accuracy_and_scores(preds, labels):
            assert preds.shape == labels.shape
            correct = preds == labels
            accuracy = correct.astype(np.float32).mean()
            # print(class_info)
            # print(labels)
            # print(preds.shape)
            # print(preds[0])
            scores = [
                mir_eval.key.weighted_score(
                    id_to_label[ref_key], id_to_label[est_key]
                )
                for ref_key, est_key in zip(labels, preds)
            ]
            # scores = [
            #     mir_eval.key.weighted_score(
            #         class_info[ref_key], class_info[est_key]
            #     )
            #     for ref_key, est_key in zip(labels, preds)
            # ]
            return accuracy, np.mean(scores)
        
        log_dict = {}
        for prefix, preds, labels in comparisons:
            accuracy, score = _compute_accuracy_and_scores(preds, labels)
            log_dict [f"{prefix}_accuracy"] = accuracy
            log_dict [f"{prefix}_score"] = score

        best_strategy_name = None
        best_score = float("-inf")

        for strategy_name in ensemble_strategy_to_song_preds.keys():
            score = log_dict[f"ensemble_{strategy_name}_score"]
            if score > best_score:
                best_strategy_name = strategy_name
                best_score = score
        log_dict[f"best_ensemble_accuracy"] = log_dict[f"ensemble_{best_strategy_name}_accuracy"]
        log_dict[f"best_ensemble_score"] = log_dict[f"ensemble_{best_strategy_name}_score"]
        log_dict["best_ensemble_strategy"] = best_strategy_name
        
        return log_dict

    def get_loss(self):
        if self.cfg.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.cfg.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        elif self.cfg.task_type == "regression":
            return nn.MSELoss()
        else:
            raise NotImplementedError(f"get_loss() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            if self.cfg.task_type == 'multilabel':
                setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                                                            task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs))
                self.all_metrics.add('ap')

                setattr(self, f"{split}_aucroc", torchmetrics.AUROC(
                                                            task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs))
                self.all_metrics.add('aucroc')

                setattr(self, f"{split}_f1", torchmetrics.F1Score(
                                                            task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs,
                                                            average='macro'))
                self.all_metrics.add('f1')

            elif self.cfg.task_type == 'multiclass':
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            task=self.cfg.task_type,
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('acc')

                setattr(self, f"{split}_prec", torchmetrics.Precision(
                                                            task=self.cfg.task_type,
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('prec')

            elif self.cfg.task_type == 'regression':
                # r2 score
                setattr(self, f"{split}_r2", torchmetrics.R2Score(num_outputs=2, multioutput='uniform_average'))
                self.all_metrics.add('r2')
                setattr(self, f"{split}_arousal_r2", torchmetrics.R2Score(num_outputs=1))
                self.all_metrics.add('arousal_r2')
                setattr(self, f"{split}_valence_r2", torchmetrics.R2Score(num_outputs=1))
                self.all_metrics.add('valence_r2')
            else:
                raise NotImplementedError(f"init_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
    
    def compute_metrics(self, split, y, y_pred, loss):
        out = {f"{split}_loss": loss}
        if self.cfg.task_type != 'regression':
            y = y.int()
            if self.cfg.task_type == 'multilabel':
                y_pred = torch.sigmoid(y_pred)
            elif self.cfg.task_type == 'multiclass':
                y_pred = torch.softmax(y_pred, dim=1)
            else:
                raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
        
        for metric_name in self.all_metrics:
            metric = getattr(self, f"{split}_{metric_name}")
            if "arousal_r2" in metric_name:
                out[f"{split}_{metric_name}"] = metric(y_pred[:, 0], y[:, 0])
            elif "valence_r2" in metric_name:
                out[f"{split}_{metric_name}"] = metric(y_pred[:, 1], y[:, 1])
            else:
                metric(y_pred, y)
            out[f"{split}_{metric_name}"] = metric
        return out


class ProberForBertSeqLabel(MLPProberBase):
    """MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec)
    This class supports learnable weighted sum over different layers of features.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = HuBERTFeature(cfg.pre_trained_folder, cfg.target_sr)
        if self.cfg.layer == "all":
            # use learned weights to aggregate features
            self.aggregator = nn.Parameter(torch.randn((self.cfg.n_tranformer_layer, 1, 1, 1)))
    
    def forward(self, x):
        input1 = self.model.process_wav(x).to(x.device)  # [batch_size, 160000]
        padding = torch.zeros(input1.shape[0], 400, device=input1.device)  # [batch_size, 2, hidden_dim]
        input1 = torch.cat((input1, padding), dim=1)     # [batch_size. 160400]

        if self.cfg.layer == "all":
            x = self.model(input1, layer=None, reduction="none")[1:]  # [12, batch_size, seq_length (249), hidden_dim]
            x = (F.softmax(self.aggregator, dim=0) * x).sum(dim=0)  # [batch_size, seq_length (249), hidden_dim]
        else:
            x = self.model(input1, layer=int(self.cfg.layer), reduction="none")  # [batch_size, seq_length (249), hidden_dim]

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = self.dropout(x)
            x = F.relu(x)
        
        output = self.output(x)
        return output
    
    def log_weights(self):
        weights = F.softmax(self.aggregator, dim=1).squeeze().detach().cpu().numpy()
        log_dict = {f"layer_{i+1}_weight": w.item() for i, w in enumerate(weights)}
        return log_dict

    def training_epoch_end(self, outputs):
        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict = self.log_weights()
                self.log_dict(log_dict, prog_bar=False)
    
    def test_epoch_end(self, outputs):
        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict = self.log_weights()
                self.log_dict(log_dict, prog_bar=False)

    def get_loss(self):
        if self.cfg.task_type == "multilabel":
            
            return nn.BCEWithLogitsLoss(weight=self.loss_weight)
            # return bce
        elif self.cfg.task_type == "multiclass":
            return nn.CrossEntropyLoss(weight=self.loss_weight)
        elif self.cfg.task_type == "regression":
            return nn.MSELoss()
        else:
            raise NotImplementedError(f"get_loss() of dataset {self.cfg.dataset} not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            if self.cfg.task_type == 'multilabel':
                setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                                                            task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs))
                self.all_metrics.add('ap')

                setattr(self, f"{split}_f1", torchmetrics.F1Score(
                                                            task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs,
                                                            average='macro'))
                self.all_metrics.add('f1')

                setattr(self, f"{split}_binary_ap", torchmetrics.AveragePrecision(
                                                            task="binary", average="macro"))
                self.all_metrics.add('binary_ap')
                
                if self.cfg.dataset == "GTZANRhythm":
                    setattr(self, f"{split}_beat_f", BCEBeatFMeasure(label_freq=self.cfg.target_sr/320,
                                                                  metric_type="beat"))
                    self.all_metrics.add("beat_f")
                    setattr(self, f"{split}_downbeat_f", BCEBeatFMeasure(label_freq=self.cfg.target_sr/320,
                                                                  metric_type="downbeat"))
                    self.all_metrics.add("downbeat_f")
                    setattr(self, f"{split}_meanbeat_f", BCEBeatFMeasure(label_freq=self.cfg.target_sr/320,
                                                                  metric_type="mean"))
                    self.all_metrics.add("meanbeat_f")

            else: # multiclass
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(task=self.cfg.task_type, 
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('acc')

                setattr(self, f"{split}_prec", torchmetrics.Precision(task=self.cfg.task_type, 
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('prec')
                
                # setattr(self, f"{split}_recall", torchmetrics.Recall(task=self.cfg.task_type, 
                #                                             num_classes=self.cfg.num_outputs))
                # self.all_metrics.add('recall')
                
                # setattr(self, f"{split}_f1", torchmetrics.F1Score(
                #                                             task=self.cfg.task_type,
                #                                             num_classes=self.cfg.num_outputs))
                # self.all_metrics.add('f1')
                
                setattr(self, f"{split}_beat_f", BeatFMeasure(label_freq=self.cfg.target_sr/320
                                                            ))
                self.all_metrics.add("beat_f")
    
    def compute_metrics(self, split, y, y_pred, loss):  # TODO: flatten the y_pred and y and use tordchmetrics
        out = {f"{split}_loss": loss}
        

        if self.cfg.task_type != 'regression':
            y = y.int()
            if self.cfg.task_type == 'multilabel':
                y_pred = torch.sigmoid(y_pred)
                y = torch.flatten(y, end_dim=1)
                y_pred = torch.flatten(y_pred, end_dim=1)
            elif self.cfg.task_type == 'multiclass':
                y_pred = torch.softmax(y_pred, dim=1)
            else:
                raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")

        for metric_name in self.all_metrics:
            metric = getattr(self, f"{split}_{metric_name}")
            
            if metric_name == 'binary_ap':
                metric(y_pred.flatten(), y.flatten())
            else:
                metric(y_pred, y)

            out[f"{split}_{metric_name}"] = metric
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        if self.cfg.task_type == "multiclass":
            y_pred = y_pred.permute(0,2,1)
        loss = self.loss(y_pred, y)
        if batch_idx % self.cfg.logging_steps == 0:
            with torch.no_grad():
                log_dict = self.compute_metrics('train', y, y_pred, loss)
                self.log_dict(log_dict, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        if self.cfg.task_type == "multiclass":
            y_pred = y_pred.permute(0,2,1)
        loss = self.loss(y_pred, y)
        log_dict = self.compute_metrics('valid', y, y_pred, loss)
        self.log_dict(log_dict, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        if self.cfg.task_type == "multiclass":
            y_pred = y_pred.permute(0,2,1)
        loss = self.loss(y_pred, y)
        log_dict = self.compute_metrics('test', y, y_pred, loss)
        self.log_dict(log_dict, prog_bar=True)


def get_model(args):
    if args.dataset in ["MTT", "MAESTRO", "MTGGenre", "MTGInstrument", "MTGMood", "MTGTop50"]:  # TODO: change here to  add pipeline
        args.task_type = "multilabel"
    elif args.dataset in ["GTZAN", "GS", "NSynthI", "NSynthP", "GTZANRhythm"]:
        args.task_type = "multiclass"
    elif args.dataset in ["EMO"]:
        args.task_type = "regression"
    else:
        raise NotImplementedError(f"get_prober() of dataset {args.dataset} not implemented")

    if args.dataset in ["MTT", "GTZAN", "GS", "EMO", "NSynthI", "NSynthP", "MTGGenre", "MTGInstrument", "MTGMood", "MTGTop50"]:  # TODO: change here to  add pipeline
        # if args.fine_tune:
        #     return FinetunerForBertCLS(args)
        # else:
        return ProberForBertCLS(args)
    elif args.dataset in ["MAESTRO", "GTZANRhythm"]:
        return ProberForBertSeqLabel(args)
    else:
        raise NotImplementedError(f"get_prober() of dataset {args.dataset} not implemented")
    




def main(args):
    # args.hidden_layer_sizes = eval(args.hidden_layer_sizes)
    pl.seed_everything(1234)
    logger = get_logger(args, train_type='probe')
    model = get_model(args)
    train_loader, valid_loader, test_loader = get_dataloaders(args, dataset_type='feature' if args.feature_dir else 'audio')
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, 
                                            logger=logger, 
                                            callbacks=callbacks, 
                                            fast_dev_run=args.debug)
    if args.eval_only:
        assert args.eval_ckpt_path, "must provide a checkpoint path for evaluation"
        assert args.strategy is None, "only support single device evaluation for now"
        trainer.validate(dataloaders=valid_loader, ckpt_path=args.eval_ckpt_path)
        trainer.test(dataloaders=test_loader, ckpt_path=args.eval_ckpt_path)
        return

    trainer.tune(model, train_loader, valid_loader)
    trainer.model.save_hyperparameters()
    trainer.fit(model, train_loader, valid_loader)
    if args.debug: return

    best_ckpt_path = trainer.checkpoint_callback.best_model_path

    # force single gpu test to avoid error
    if args.strategy is not None: 
        assert args.strategy == "ddp", "only support ddp strategy for now, other strategies may not get the right numbers"
        torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        args.devices = 1
        args.num_nodes = 1
        args.strategy = None
        trainer = pl.Trainer.from_argparse_args(args, 
                                            logger=logger
                                            )
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=best_ckpt_path)

        # does it really save the best model?
        if args.save_best_to is not None: trainer.save_checkpoint(args.save_best_to)