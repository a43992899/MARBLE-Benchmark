import argparse
from copy import deepcopy

import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from pytorch_lightning.utilities import parsing as plparsing
from pytorch_lightning.utilities.parsing import lightning_getattr

import benchmark as bench
from benchmark.utils.config_utils import to_dict
from .poolings import *

class MLPProberBase(pl.LightningModule):
    """Base MLP prober
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        selected_cfg = self.select_cfg()
        self.save_hyperparameters(selected_cfg)
        
        d = self.hparams.num_features
        self.hidden_layer_sizes = eval(self.hparams.hidden_layer_sizes)
        self.num_layers = len(self.hidden_layer_sizes)
        for i, ld in enumerate(self.hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld
        self.output = nn.Linear(d, self.hparams.num_outputs)
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)
        self.loss = self.get_loss() # can return a dict of losses
        self.init_metrics()
        self.monitor_mode = 'min' if 'loss' in self.hparams.monitor else 'max'
    
    def select_cfg(self):
        """ Select a subset of cfg and save them in hparams.

        This allows easier access to the config when probing, and also allows
        to logging and checkpointing the config.
        
        TODO: 
        dataset.dataset 改成 dataset.name
        处理不同pretrain的预训练参数不一致的问题
        checkpoint里有一些参数未暴露出来,比如save_top_k,save_last,save_weights_only
        sequence task, finetune的时候, 有些args未支持, 看deprecated__main__.py
        add resume from checkpoint
        """
        _cfg = self.cfg
        cfg = argparse.Namespace()
        # # backup full config into checkpoint and log
        # cfg._backup_cfg = _cfg
        self.select_runtime_cfg(cfg, _cfg)
        self.select_dataset_cfg(cfg, _cfg)
        self.select_upstream_cfg(cfg, _cfg)
        self.select_downstream_cfg(cfg, _cfg)
        self.select_training_cfg(cfg, _cfg)
        self.select_logger_cfg(cfg, _cfg)
        return cfg
    
    def select_runtime_cfg(self, cfg, _cfg):
        cfg.task_type = _cfg._runtime.task_type

    def select_dataset_cfg(self, cfg, _cfg):
        cfg.dataset = _cfg.dataset.dataset
        cfg.metadata_dir = _cfg.dataset.metadata_dir
        cfg.input_type = _cfg.dataset.input_type
        cfg.input_dir = _cfg.dataset.input_dir
    
    def select_upstream_cfg(self, cfg, _cfg):
        if hasattr(_cfg.dataset, 'pre_extract'):
            pretrained_extractor = _cfg.dataset.pre_extract.feature_extractor.pretrain
        else:
            pretrained_extractor = _cfg.model.feature_extractor.pretrain
        cfg.pretrained_extractor = pretrained_extractor 
        cfg.pre_trained_model_name = pretrained_extractor.name
        cfg.num_features = pretrained_extractor.num_features
        cfg.target_sr = pretrained_extractor.target_sr
        # check if it is a transformer
        if hasattr(pretrained_extractor, 'n_tranformer_layer'):
            cfg.n_tranformer_layer = pretrained_extractor.n_tranformer_layer
            cfg.token_rate = pretrained_extractor.token_rate
    
    def select_downstream_cfg(self, cfg, _cfg):
        downstream_structure_components = _cfg.model.downstream_structure.components
        for component in downstream_structure_components:
            if component.name == 'feature_selector':
                cfg.layer = component.layer
                cfg.normalized_weight_sum = component.normalized_weight_sum
            if component.name == 'mlp':
                cfg.hidden_layer_sizes = str(component.hidden_layer_sizes)
                cfg.dropout_p = component.dropout_p
                cfg.num_outputs = component.num_outputs
    
    def select_training_cfg(self, cfg, _cfg):
        cfg.auto_lr_find = _cfg.trainer.auto_lr_find
        cfg.accelerator = _cfg.trainer.accelerator
        cfg.devices = _cfg.trainer.devices
        cfg.strategy = _cfg.trainer.strategy
        cfg.precision = _cfg.trainer.precision
        cfg.accumulate_grad_batches = _cfg.trainer.accumulate_grad_batches
        cfg.max_epochs = _cfg.trainer.max_epochs
        cfg.seed = _cfg.trainer.seed
        cfg.paradigm = _cfg.trainer.paradigm

        cfg.num_workers = _cfg.dataloader.num_workers
        cfg.batch_size = _cfg.dataloader.batch_size
        cfg.optimizer = _cfg.optimizer.name
        cfg.lr = eval(_cfg.optimizer.lr) if isinstance(_cfg.optimizer.lr, str) else _cfg.optimizer.lr
        cfg.l2_weight_decay = _cfg.optimizer.l2_weight_decay

        cfg.lr_scheduler_patience = _cfg.scheduler.lr_scheduler_patience
        cfg.earlystop_patience = _cfg.scheduler.earlystop_patience

        cfg.loss_weight = _cfg.loss.loss_weight
    
    def select_logger_cfg(self, cfg, _cfg):
        cfg.wandb_sweep = _cfg.logger.wandb_sweep
        cfg.wandb_off = _cfg.logger.wandb_off
        cfg.monitor = _cfg.logger.monitor

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
        self.update_metrics('train', y, y_pred)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('valid', y, y_pred)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('test', y, y_pred)
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        lr = lightning_getattr(self, 'lr')
        Optimizer = eval(self.hparams.optimizer)
        optimizer = Optimizer(self.parameters(), lr=lr, weight_decay=self.hparams.l2_weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.monitor_mode, factor=0.5, patience=self.hparams.lr_scheduler_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.hparams.monitor}
    
    def get_loss(self):
        raise NotImplementedError(f"get_loss() not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        raise NotImplementedError(f"init_metrics() not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        raise NotImplementedError(f"update_metrics() not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def log_metrics(self, split):
        raise NotImplementedError(f"log_metrics() not implemented in {self.__class__.__name__}")


class ProberForBertUtterCLS(MLPProberBase):
    """MLP Prober for utterance-level classification tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Note that this class assumes that the features are pre-extracted and (chunk) time averaged.
    This class supports learnable weighted sum over different layers of features.

    You should reimplement get_loss(), init_metrics(), update_metrics(), log_metrics() if your task is
    diferent from standard classification tasks, or you want to do multitask learning.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_aggregator()
        self.init_scatter_mean()
    
    def forward(self, x):
        """
        x: (B, L, T, H)
        T=#chunks, can be 1 or several chunks
        """
        if self.hparams.layer == "all":
            if isinstance(self.aggregator, nn.Conv1d):
                x = self.aggregator(x).squeeze()
            else:
                weights = F.softmax(self.aggregator, dim=1)
                x = (x * weights).sum(dim=1)

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.output(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('train', y, y_pred)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, index = batch # x: [batch_size, n_layer_features, hidden_dim]
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, batch_size=y.shape[0], prog_bar=True, sync_dist=True)
        self.update_metrics('valid', y, y_pred)
    
    def test_step(self, batch, batch_idx):
        x, y, index = batch
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, batch_size=y.shape[0], prog_bar=True, sync_dist=True)
        self.update_metrics('test', y, y_pred)

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            self.log_metrics('train')
            if self.hparams.layer == "all":
                if not isinstance(self.aggregator, nn.Conv1d):
                    log_dict = self.log_weights('train')
                    self.log_dict(log_dict, sync_dist=True)
    
    def validation_epoch_end(self, outputs):
        self.log_metrics('valid')
    
    def test_epoch_end(self, outputs):
        self.log_metrics('test')
        if self.hparams.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict = self.log_weights('test')
                self.log_dict(log_dict)

    def get_loss(self):
        if self.hparams.task_type == "multilabel":
            return nn.BCEWithLogitsLoss()
        elif self.hparams.task_type == "multiclass":
            return nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"get_loss() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    def init_metrics(self):
        """supports standard classification metrics"""
        self.all_metrics = set()

        for split in ['train', 'valid', 'test']:
            if self.hparams.task_type == 'multilabel':
                setattr(self, f"{split}_ap", torchmetrics.AveragePrecision(
                                                            task=self.hparams.task_type,
                                                            num_labels=self.hparams.num_outputs))
                self.all_metrics.add('ap')

                setattr(self, f"{split}_aucroc", torchmetrics.AUROC(
                                                            task=self.hparams.task_type,
                                                            num_labels=self.hparams.num_outputs))
                self.all_metrics.add('aucroc')

                setattr(self, f"{split}_f1", torchmetrics.F1Score(
                                                            task=self.hparams.task_type,
                                                            num_labels=self.hparams.num_outputs,
                                                            average='macro'))
                self.all_metrics.add('f1')

            elif self.hparams.task_type == 'multiclass':
                setattr(self, f"{split}_acc", torchmetrics.Accuracy(
                                                            task=self.hparams.task_type,
                                                            num_classes=self.hparams.num_outputs))
                self.all_metrics.add('acc')

                setattr(self, f"{split}_prec", torchmetrics.Precision(
                                                            task=self.hparams.task_type,
                                                            num_classes=self.hparams.num_outputs))
                self.all_metrics.add('prec')

            else:
                raise NotImplementedError(f"init_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        """update metrics during train/valid/test step"""
        y = y.int()
        if self.hparams.task_type == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
        elif self.hparams.task_type == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
        y_pred = y_pred.detach()

        if self.hparams.task_type == 'multilabel':
            getattr(self, f"{split}_ap").update(y_pred, y)
            getattr(self, f"{split}_aucroc").update(y_pred, y)
            getattr(self, f"{split}_f1").update(y_pred, y)
        elif self.hparams.task_type == 'multiclass':
            getattr(self, f"{split}_acc").update(y_pred, y)
            getattr(self, f"{split}_prec").update(y_pred, y)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def log_metrics(self, split):
        """log metrics on epoch end, manually reset
        """
        if self.hparams.task_type == 'multilabel':
            self.log(f"{split}_ap", getattr(self, f"{split}_ap").compute(), sync_dist=True)
            getattr(self, f"{split}_ap").reset()
            self.log(f"{split}_aucroc", getattr(self, f"{split}_aucroc").compute(), sync_dist=True)
            getattr(self, f"{split}_aucroc").reset()
            self.log(f"{split}_f1", getattr(self, f"{split}_f1").compute(), sync_dist=True)
            getattr(self, f"{split}_f1").reset()
        elif self.hparams.task_type == 'multiclass':
            self.log(f"{split}_acc", getattr(self, f"{split}_acc").compute(), sync_dist=True)
            getattr(self, f"{split}_acc").reset()
            self.log(f"{split}_prec", getattr(self, f"{split}_prec").compute(), sync_dist=True)
            getattr(self, f"{split}_prec").reset()
        else:
            raise NotImplementedError(f"log_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    @torch.no_grad()
    def log_weights(self, split):
        """transformer layer weighted ensemble"""
        weights = F.softmax(self.aggregator, dim=1).squeeze().detach().cpu().numpy()
        log_dict = {f"{split}_layer_{i+1}_weight": w.item() for i, w in enumerate(weights)}
        return log_dict
    
    def init_aggregator(self):
        """Initialize the aggregator for weighted sum over different layers of features
        """
        if self.hparams.layer == "all":
            # use learned weights to aggregate features
            if self.hparams.normalized_weight_sum:
                self.aggregator = nn.Parameter(torch.randn((1, self.hparams.n_tranformer_layer, 1)))
            else:
                self.aggregator = nn.Conv1d(in_channels=self.hparams.n_tranformer_layer, out_channels=1, kernel_size=1)
    
    def init_scatter_mean(self):
        try:
            from torch_scatter import scatter_mean
            self.scatter_mean = scatter_mean
            print("[Info]: torch_scatter is installed, use cuda implementation.\n")
        except ImportError:
            print("[Warning]: torch_scatter is not installed, use non-cuda implementation.\n")

            def scatter_mean(y_pred, index, dim=0):
                """non-cuda implementation of torch_scatter.scatter_mean.

                you can use torch_scatter.scatter_mean if you install torch_scatter
                it is used for chunk level prediction ensemble to obtain song level prediction
                """
                new_y_pred = torch.zeros((index.unique().shape[0], y_pred.shape[1]), device=y_pred.device)
                for i in index.unique():
                    new_y_pred[i] = y_pred[index == i].mean(dim=dim)
                return new_y_pred
            
            self.scatter_mean = scatter_mean


class ProberForBertSeqLabel(ProberForBertUtterCLS):
    """MLP Prober for sequence labeling tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Sequence labeling tasks are token level classification tasks, in MIR, it is usually used for
    onset detection, note transcription, etc.

    This class supports learnable weighted sum over different layers of features.

    TODO: implement MAESTRO_prober
    TODO: enable biLSTM
    TODO: fix refresh rate
    TODO: support define torchmetrics
    TODO: support MAESTRO fast dataset
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = self.init_extractor()
    
    def forward(self, x):
        # TODO: this is slow as the data is transferred between GPU and CPU multiple times
        x = self.bert.process_wav(x).to(x.device)  # [B, T]
        padding = torch.zeros(x.shape[0], 320, device=x.device) # this is not common for every model
        x = torch.cat((x, padding), dim=1)

        if self.hparams.layer == "all":
            with torch.no_grad():
                x = self.bert(x, layer=None, reduction="none")[1:]  # [12, batch_size, seq_length (249), hidden_dim]
            x = (F.softmax(self.aggregator, dim=0) * x).sum(dim=0)  # [batch_size, seq_length (249), hidden_dim]
        else:
            with torch.no_grad():
                x = self.bert(x, layer=int(self.hparams.layer), reduction="none")  # [batch_size, seq_length (249), hidden_dim]

        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = self.dropout(x)
            x = F.relu(x)
        
        output = self.output(x)
        return output
    
    def select_dataset_cfg(self, cfg, _cfg):
        super().select_dataset_cfg(cfg, _cfg)
    
    def select_upstream_cfg(self, cfg, _cfg):
        super().select_upstream_cfg(cfg, _cfg)
        pre_trained_folder = getattr(
            cfg.pretrained_extractor, 
            'pre_trained_folder', 
            None
        )
        huggingface_model_name = getattr(
            cfg.pretrained_extractor, 
            'huggingface_model_name', 
            None
        )
        if pre_trained_folder is not None:
            cfg.pre_trained_folder = pre_trained_folder
        else:
            cfg.pre_trained_folder = huggingface_model_name
        
        cfg.force_half = _cfg.model.feature_extractor.force_half
        cfg.reduction = _cfg.model.feature_extractor.reduction
        cfg.processor_normalize = _cfg.model.feature_extractor.pretrain.processor_normalize
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('valid', y, y_pred)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('test', y, y_pred)
    
    @torch.no_grad()
    def update_metrics(self, split, y, y_pred):
        """update metrics during train/valid/test step"""
        y = y.int() # [B, T, C], C is the number of classes
        if self.hparams.task_type == 'multilabel':
            y_pred = torch.sigmoid(y_pred)
        elif self.hparams.task_type == 'multiclass':
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
        y = torch.flatten(y, end_dim=1) # [BxT, C]
        y_pred = torch.flatten(y_pred, end_dim=1)
        y_pred = y_pred.detach()

        if self.hparams.task_type == 'multilabel':
            getattr(self, f"{split}_ap").update(y_pred, y)
            getattr(self, f"{split}_aucroc").update(y_pred, y)
            getattr(self, f"{split}_f1").update(y_pred, y)
        elif self.hparams.task_type == 'multiclass':
            getattr(self, f"{split}_acc").update(y_pred, y)
            getattr(self, f"{split}_prec").update(y_pred, y)
        else:
            raise NotImplementedError(f"compute_metrics() of {self.hparams.task_type} not implemented in {self.__class__.__name__}")
    
    def init_aggregator(self):
        if self.hparams.layer == "all":
            # use learned weights to aggregate features
            if self.hparams.normalized_weight_sum:
                self.aggregator = nn.Parameter(torch.randn((self.hparams.n_tranformer_layer, 1, 1, 1)))
            else:
                self.aggregator = nn.Conv1d(in_channels=self.hparams.n_tranformer_layer, out_channels=1, kernel_size=1)
    
    def init_scatter_mean(self):
        # skip init scatter_mean
        pass

    def init_extractor(self):
        """Return a pretrain bert extractor
        """
        model_name = self.hparams.pre_trained_model_name
        FeatureExtractor = eval(
            bench.NAME_TO_PRETRAIN_CLASS[model_name]
        )
        feature_extractor = FeatureExtractor(
            self.hparams.pre_trained_folder,
            self.hparams.target_sr,
            self.hparams.force_half,
            processor_normalize=self.hparams.processor_normalize,
        )
        return feature_extractor


class ProberForBertSeqCLS(ProberForBertSeqLabel):
    """MLP Prober for sequence classification tasks with BERT like 12 layer features (HuBERT, Data2vec).

    Sequence classification tasks are utterance level classification tasks. in MIR, it is usually used for
    audio tagging/classification, etc.

    It does not assume time avg features, but extracts features on the fly, such that more sophisticated
    feature aggregation can be done, e.g. GeM pooling, lstm aggregation, etc.
    This class supports learnable weighted sum over different layers of features.

    TODO: complete this class
    """
    def __init__(self, cfg, model):
        super().__init__(cfg)
        # TODO: support different pooling

