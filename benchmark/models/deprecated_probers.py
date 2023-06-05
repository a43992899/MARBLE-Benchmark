import os, sys

import numpy as np
import wandb
import argparse
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import parsing as plparsing
from pytorch_lightning.utilities.parsing import lightning_getattr

import benchmark as bench
from benchmark.extract_hubert_features import HuBERTFeature
from benchmark.utils.get_dataloader import get_dataloaders
from benchmark.utils.get_logger import get_logger
from benchmark.utils.get_callbacks import get_callbacks 
from benchmark.tasks.GS.utils import predict_result_ensemble as gs_ensemble

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
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.compute_metrics('train', y, y_pred, compute=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.compute_metrics('valid', y, y_pred, compute=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.compute_metrics('test', y, y_pred, compute=False)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        lr = lightning_getattr(self, 'lr')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.cfg.l2_weight_decay)
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
        
        if self.cfg.dataset=="GS":
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
    
    def scatter_mean(self, y_pred, index, dim=0):
        new_y_pred = torch.zeros((index.unique().shape[0], y_pred.shape[1]), device=y_pred.device)
        for i in index.unique():
            new_y_pred[i] = y_pred[index == i].mean(dim=dim)
        return new_y_pred
    
    def log_my_dict(self, my_dict):
        for k, v in my_dict.items():
            self.log(k, v, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.cfg.dataset=="GS":
                x, y, meta_idx, class_in_str, index = batch
            else:
                # x, y = batch  # x: [batch_size, n_layer_features, hidden_dim]
                x, y, index = batch
            y_pred = self(x)  # [batch_size, n_class]
            y_pred = self.scatter_mean(y_pred, index, dim=0)
            loss = self.loss(y_pred, y)
            self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
            self.compute_metrics('valid', y, y_pred, compute=False)

        if self.cfg.dataset=="GS":
            return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), class_in_str
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.cfg.dataset=="GS":
                x, y, meta_idx, class_in_str, index = batch
            else:
                x, y, index = batch
            y_pred = self(x)  # [batch_size, n_class]
            y_pred = self.scatter_mean(y_pred, index, dim=0)
            loss = self.loss(y_pred, y)
            self.log('test_loss', loss, prog_bar=True, sync_dist=True)
            self.compute_metrics('test', y, y_pred, compute=False)

        if self.cfg.dataset=="GS":
            return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), class_in_str
    
    def log_weights(self):
        weights = F.softmax(self.aggregator, dim=1).squeeze().detach().cpu().numpy()
        log_dict = {f"layer_{i+1}_weight": w.item() for i, w in enumerate(weights)}
        return log_dict

    def training_epoch_end(self, outputs):
        log_dict = self.compute_metrics('train', compute=True)
        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                with torch.no_grad():
                    log_dict = self.log_weights()
                    self.log_my_dict(log_dict)
    
    def validation_epoch_end(self, outputs):
        log_dict = self.compute_metrics('valid', compute=True)

        # test ensemble will sooon be deprecated. 
        # simply extract chunk based features and will automatically do ensemble
        # TODO: reprocess GS
        if self.cfg.dataset=="GS":
            with torch.no_grad():
                log_ensemble_results = gs_ensemble(outputs, self.device) # [(strategy_name, predicted_label, label), ...]
                best_strategy_name = log_ensemble_results.pop("best_ensemble_strategy")
                if log_ensemble_results["best_ensemble_score"] > self.valid_best_ensemble_score:
                    self.valid_best_ensemble_score = log_ensemble_results["best_ensemble_score"]
                    self.valid_best_ensemble_strategy = best_strategy_name
                if not self.cfg.wandb_off:
                    self.logger.experiment.log( {"valid_best_ensemble_strategy": best_strategy_name}) # equals to wandb.log()
                log_ensemble_results = dict(("valid_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
                log_dict.update(log_ensemble_results)

        self.log_my_dict(log_dict)

        # self.log('valid_aucroc', self.valid_aucroc, sync_dist=True)
        # self.log('valid_ap', self.valid_ap, sync_dist=True)
        # self.log('valid_f1', self.valid_f1, sync_dist=True)

        # for metric_name in self.all_metrics:
        #     self.log(f"valid_{metric_name}", getattr(self, f"valid_{metric_name}"), sync_dist=True)

        # for metric_name in self.all_metrics:
        #     if 'aucroc' in metric_name:
        #         self.log('valid_aucroc', self.valid_aucroc, sync_dist=True)
        #     elif 'ap' in metric_name:
        #         self.log('valid_ap', self.valid_ap, sync_dist=True)
        #     elif 'f1' in metric_name:
        #         self.log('valid_f1', self.valid_f1, sync_dist=True)
    
    def test_epoch_end(self, outputs):
        '''
        @yizhilll: https://github.com/Lightning-AI/lightning/discussions/9914
        the test_step outputs could be aggregated if using DataParrallel, we don't support this yet.
        outputs
        '''
        log_dict = self.compute_metrics('test', compute=True)
        with torch.no_grad():
            if self.cfg.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict.update(self.log_weights())
            
            # test ensemble will sooon be deprecated. 
            # simply extract chunk based features and will automatically do ensemble
            # TODO: reprocess GS
            if self.cfg.dataset=="GS":
                log_ensemble_results = gs_ensemble(outputs, self.device) # [(strategy_name, predicted_label, label), ...]
                # additional handling the string log
                log_ensemble_results.pop("best_ensemble_strategy") # no need for test_best_strategy_name
                if self.valid_best_ensemble_strategy is not None:
                    log_ensemble_results["ensemble_score_val-select"] = log_ensemble_results[f"ensemble_{self.valid_best_ensemble_strategy}_score"]
                # self.logger.experiment.config["test_best_ensemble_strategy"] = log_ensemble_results.pop("best_ensemble_strategy")
                log_ensemble_results = dict(("test_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
                log_dict.update(log_ensemble_results)
        self.log_my_dict(log_dict)

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
    
    def compute_metrics(self, split, y=None, y_pred=None, compute=False):
        if y is not None:
            if self.cfg.task_type != 'regression':
                y = y.int()
                if self.cfg.task_type == 'multilabel':
                    y_pred = torch.sigmoid(y_pred)
                elif self.cfg.task_type == 'multiclass':
                    y_pred = torch.softmax(y_pred, dim=1)
                else:
                    raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
        
        out = {}
        for metric_name in self.all_metrics:
            metric = getattr(self, f"{split}_{metric_name}")

            if y is not None:
                if "arousal_r2" in metric_name:
                    metric.update(y_pred[:, 0].detach(), y[:, 0])
                elif "valence_r2" in metric_name:
                    metric.update(y_pred[:, 1].detach(), y[:, 1])
                else:
                    metric.update(y_pred.detach(), y)
                
            if compute:
                out[f"{split}_{metric_name}"] = metric

        return out


class ProberForBertSeqLabel(MLPProberBase):
    """MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec)
    This class supports learnable weighted sum over different layers of features.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = HuBERTFeature(cfg.pre_trained_folder, cfg.target_sr)
        self.model.eval()
        if self.cfg.layer == "all":
            # use learned weights to aggregate features
            self.aggregator = nn.Parameter(torch.randn((self.cfg.n_tranformer_layer, 1, 1, 1)))
        # TODO: enable biLSTM
        # TODO: fix ddp
        # TODO: fix refresh rate
        # TODO: support define torchmetrics
        # TODO: support MAESTRO fast dataset
    
    def forward(self, x):
        x = self.model.process_wav(x).to(x.device)  # [batch_size, 160000]
        padding = torch.zeros(x.shape[0], 320, device=x.device)  # [batch_size, 2, hidden_dim]
        x = torch.cat((x, padding), dim=1)     # [batch_size. 160400]

        if self.cfg.layer == "all":
            with torch.no_grad():
                x = self.model(x, layer=None, reduction="none")[1:]  # [12, batch_size, seq_length (249), hidden_dim]
            x = (F.softmax(self.aggregator, dim=0) * x).sum(dim=0)  # [batch_size, seq_length (249), hidden_dim]
        else:
            with torch.no_grad():
                x = self.model(x, layer=int(self.cfg.layer), reduction="none")  # [batch_size, seq_length (249), hidden_dim]

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
        log_dict = self.compute_metrics('test')

        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict.update(self.log_weights())
        self.log_dict(log_dict)


    def get_loss(self):
        if self.cfg.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.cfg.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        elif self.cfg.task_type == "regression":
            return nn.MSELoss()
        else:
            raise NotImplementedError(f"get_loss() of dataset {self.cfg.dataset} not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            if self.cfg.task_type == 'multilabel':
                # setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                #                                             task=self.cfg.task_type,
                #                                             num_labels=self.cfg.num_outputs))
                # self.all_metrics.add('ap')

                # setattr(self, f"{split}_f1", torchmetrics.F1Score(
                #                                             task=self.cfg.task_type,
                #                                             num_labels=self.cfg.num_outputs,
                #                                             average='macro'))
                # self.all_metrics.add('f1')

                setattr(self, f"{split}_binary_ap", torchmetrics.AveragePrecision(
                                                            task="binary", average="macro",
                                                            thresholds=40
                                                            ))
                self.all_metrics.add('binary_ap')

            else: # multiclass
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('acc')

                setattr(self, f"{split}_prec", torchmetrics.Precision(
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('prec')
    
    @torch.no_grad()
    def compute_metrics(self, split, y=None, y_pred=None, loss=None):  
        out = {}
        if loss is not None:
            out = {f"{split}_loss": loss.item()}
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
            if loss is not None:
                if metric_name == 'binary_ap':
                    metric(y_pred.flatten().detach(), y.flatten())
                else:
                    metric(y_pred.detach(), y)
            out[f"{split}_{metric_name}"] = metric
        return out


class _ProberForBertUtterCLS(MLPProberBase):
    '''Deprecated. 
    MLP Prober for utterance-level classification tasks with BERT like 12 layer features (HuBERT, Data2vec).

    This class supports learnable weighted sum over different layers of features.
    Note that we assume the features are pre-extracted and time averaged (since these audio berts do not have CLS token).
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.cfg.layer == "all":
            # use learned weights to aggregate features
            if self.cfg.normalized_weight_sum:
                self.aggregator = nn.Parameter(torch.randn((1, self.cfg.n_tranformer_layer, 1)))
            else:
                self.aggregator = nn.Conv1d(in_channels=self.cfg.n_tranformer_layer, out_channels=1, kernel_size=1)
    
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
    
    def scatter_mean(self, y_pred, index, dim=0):
        new_y_pred = torch.zeros((index.unique().shape[0], y_pred.shape[1]), device=y_pred.device)
        for i in index.unique():
            new_y_pred[i] = y_pred[index == i].mean(dim=dim)
        return new_y_pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self._update_metrics('train', y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, index = batch # x: [batch_size, n_layer_features, hidden_dim]
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        # self.update_metrics('valid', y, y_pred)
        # self.log_metrics('valid')
        self._update_metrics('valid', y, y_pred)
    
    def test_step(self, batch, batch_idx):
        x, y, index = batch
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        # self.update_metrics('test', y, y_pred)
        # self.log_metrics('test')
        self._update_metrics('test', y, y_pred)
    
    @torch.no_grad()
    def log_weights(self):
        # transformer layer weighted ensemble
        weights = F.softmax(self.aggregator, dim=1).squeeze().detach().cpu().numpy()
        log_dict = {f"layer_{i+1}_weight": w.item() for i, w in enumerate(weights)}
        return log_dict

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            self._log_metrics('train')
            if self.cfg.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict = self.log_weights()
                    self.log_dict(log_dict, sync_dist=True)
    
    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            self._log_metrics('valid')
            if self.cfg.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict = self.log_weights()
                    self.log_dict(log_dict)
    
    def test_epoch_end(self, outputs):
        with torch.no_grad():
            self._log_metrics('test')
            if self.cfg.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict = self.log_weights()
                    self.log_dict(log_dict)

    def get_loss(self):
        """
        reimplement this if your task is different from standard classification tasks,
        or you want to perform multi-task learning with multiple loss functions,
        you should simply return a list of losses
        """
        if self.cfg.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.cfg.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"get_loss() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
    
    # def init_metrics(self):
    #     self.all_metrics = set()

    #     for split in ['train', 'valid', 'test']:
    #         if self.cfg.task_type == 'multilabel':
    #             setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
    #                                                         task=self.cfg.task_type,
    #                                                         num_labels=self.cfg.num_outputs))
    #             self.all_metrics.add('ap')

    #             setattr(self, f"{split}_aucroc", torchmetrics.AUROC(
    #                                                         task=self.cfg.task_type,
    #                                                         num_labels=self.cfg.num_outputs))
    #             self.all_metrics.add('aucroc')

    #             setattr(self, f"{split}_f1", torchmetrics.F1Score(
    #                                                         task=self.cfg.task_type,
    #                                                         num_labels=self.cfg.num_outputs,
    #                                                         average='macro'))
    #             self.all_metrics.add('f1')

    #         elif self.cfg.task_type == 'multiclass':
    #             setattr(self, f"{split}_acc", torchmetrics.Accuracy(
    #                                                         task=self.cfg.task_type,
    #                                                         num_classes=self.cfg.num_outputs))
    #             self.all_metrics.add('acc')

    #             setattr(self, f"{split}_prec", torchmetrics.Precision(
    #                                                         task=self.cfg.task_type,
    #                                                         num_classes=self.cfg.num_outputs))
    #             self.all_metrics.add('prec')

    #         else:
    #             raise NotImplementedError(f"init_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        y = y.int()
        if self.cfg.task_type == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
        elif self.cfg.task_type == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
        y_pred = y_pred.detach()

        if self.cfg.task_type == 'multilabel':
            getattr(self, f"{split}_ap").update(y_pred, y)
            getattr(self, f"{split}_aucroc").update(y_pred, y)
            getattr(self, f"{split}_f1").update(y_pred, y)
        elif self.cfg.task_type == 'multiclass':
            getattr(self, f"{split}_acc").update(y_pred, y)
            getattr(self, f"{split}_prec").update(y_pred, y)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def log_metrics(self, split, on_step=False, on_epoch=True):
        """ log metrics call during step
        """
        if self.cfg.task_type == 'multilabel':
            self.log(f"{split}_ap", getattr(self, f"{split}_ap").compute(), sync_dist=True, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{split}_aucroc", getattr(self, f"{split}_aucroc").compute(), sync_dist=True, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{split}_f1", getattr(self, f"{split}_f1").compute(), sync_dist=True, on_step=on_step, on_epoch=on_epoch)
        elif self.cfg.task_type == 'multiclass':
            self.log(f"{split}_acc", getattr(self, f"{split}_acc").compute(), sync_dist=True, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{split}_prec", getattr(self, f"{split}_prec").compute(), sync_dist=True, on_step=on_step, on_epoch=on_epoch)
        else:
            raise NotImplementedError(f"log_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")

    
    def init_metrics(self):
        """ init metrics with MetricCollection
        It will be deprecated as it does not support custom naming of metrics
        """
        metrics = []
        metric_name_mapper = dict()
        if self.cfg.task_type == 'multilabel':
            metrics.append(torchmetrics.AveragePrecision(task=self.cfg.task_type,
                                                            num_labels=self.cfg.num_outputs,
                                                            average='macro'))
            metrics.append(torchmetrics.AUROC(task=self.cfg.task_type,
                                                num_labels=self.cfg.num_outputs,
                                                average='macro'))
            metrics.append(torchmetrics.F1Score(task=self.cfg.task_type,
                                                num_labels=self.cfg.num_outputs,
                                                average='macro'))
        elif self.cfg.task_type == 'multiclass':
            metrics.append(torchmetrics.Accuracy(task=self.cfg.task_type,
                                                    num_classes=self.cfg.num_outputs))
            metrics.append(torchmetrics.Precision(task=self.cfg.task_type,
                                                    num_classes=self.cfg.num_outputs))
        else:
            raise NotImplementedError(f"init_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")
        
        metrics = MetricCollection(metrics)
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')

    @torch.no_grad()
    def _update_metrics(self, split, y, y_pred):
        """ update metrics with MetricCollection
        deprecated
        """
        y = y.int()
        if self.cfg.task_type == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
        elif self.cfg.task_type == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.cfg.task_type} not implemented in {self.__class__.__name__}")

        metric_of_split = getattr(self, f"{split}_metrics")
        metric_of_split.update(y_pred.detach(), y)
    
    @torch.no_grad()
    def _log_metrics(self, split):
        """ log metrics with MetricCollection
        deprecated"""
        # because of the reset, the metrics must be logged only once per epoch
        metric_of_split = getattr(self, f"{split}_metrics")
        metrics = metric_of_split.compute()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        metric_of_split.reset()


class _ProberForBertSeqLabel(MLPProberBase):
    """Deprecated. 
    MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Sequence labeling tasks are token level classification tasks, in MIR, it is usually used for
    onset detection, note transcription, etc.

    This class supports learnable weighted sum over different layers of features.
    """
    def __init__(self, cfg, model):
        super().__init__(cfg)
        self.model = model
        self.model.eval()
        if self.cfg.layer == "all":
            # use learned weights to aggregate features
            self.aggregator = nn.Parameter(torch.randn((self.cfg.n_tranformer_layer, 1, 1, 1)))
        # TODO: enable biLSTM
        # TODO: fix refresh rate
        # TODO: support define torchmetrics
        # TODO: support MAESTRO fast dataset
    
    def forward(self, x):
        x = self.model.process_wav(x).to(x.device)  # [B, T]
        padding = torch.zeros(x.shape[0], 320, device=x.device)
        x = torch.cat((x, padding), dim=1)

        if self.cfg.layer == "all":
            with torch.no_grad():
                x = self.model(x, layer=None, reduction="none")[1:]  # [12, batch_size, seq_length (249), hidden_dim]
            x = (F.softmax(self.aggregator, dim=0) * x).sum(dim=0)  # [batch_size, seq_length (249), hidden_dim]
        else:
            with torch.no_grad():
                x = self.model(x, layer=int(self.cfg.layer), reduction="none")  # [batch_size, seq_length (249), hidden_dim]

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
        log_dict = self.compute_metrics('test')

        if self.cfg.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict.update(self.log_weights())
        self.log_dict(log_dict)


    def get_loss(self):
        if self.cfg.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.cfg.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"get_loss() of dataset {self.cfg.dataset} not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            if self.cfg.task_type == 'multilabel':
                # setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                #                                             task=self.cfg.task_type,
                #                                             num_labels=self.cfg.num_outputs))
                # self.all_metrics.add('ap')

                # setattr(self, f"{split}_f1", torchmetrics.F1Score(
                #                                             task=self.cfg.task_type,
                #                                             num_labels=self.cfg.num_outputs,
                #                                             average='macro'))
                # self.all_metrics.add('f1')

                setattr(self, f"{split}_binary_ap", torchmetrics.AveragePrecision(
                                                            task="binary", average="macro",
                                                            thresholds=40
                                                            ))
                self.all_metrics.add('binary_ap')

            else: # multiclass
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('acc')

                setattr(self, f"{split}_prec", torchmetrics.Precision(
                                                            num_classes=self.cfg.num_outputs))
                self.all_metrics.add('prec')
    
    @torch.no_grad()
    def compute_metrics(self, split, y=None, y_pred=None, loss=None):  
        out = {}
        if loss is not None:
            out = {f"{split}_loss": loss.item()}
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
            if loss is not None:
                if metric_name == 'binary_ap':
                    metric(y_pred.flatten().detach(), y.flatten())
                else:
                    metric(y_pred.detach(), y)
            out[f"{split}_{metric_name}"] = metric
        return out
