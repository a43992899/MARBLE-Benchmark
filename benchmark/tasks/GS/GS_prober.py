from torch import nn

from benchmark.models.probers import ProberForBertUtterCLS
from benchmark.tasks.GS.utils import predict_result_ensemble as gs_ensemble

class GSProber(ProberForBertUtterCLS):
    '''This class adapts ensemble logits from jukemir. Will be deprecated soon.
    '''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.valid_best_ensemble_score = float("-inf")
        self.valid_best_ensemble_strategy = None
    
    def validation_step(self, batch, batch_idx):
        x, y, meta_idx, class_in_str, index = batch # x: [batch_size, n_layer_features, hidden_dim]
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('valid', y, y_pred)
        return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), class_in_str
    
    def test_step(self, batch, batch_idx):
        x, y, meta_idx, class_in_str, index = batch
        y_pred = self(x)  # [batch_size, n_class]
        y_pred = self.scatter_mean(y_pred, index, dim=0)
        loss = self.loss(y_pred, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.update_metrics('test', y, y_pred)
        return y.cpu().numpy(), y_pred.cpu().numpy(), meta_idx.cpu().numpy(), class_in_str
    
    def validation_epoch_end(self, outputs):
        self.log_metrics('valid')
        log_dict = dict()
                
        log_ensemble_results = gs_ensemble(outputs, self.device) # [(strategy_name, predicted_label, label), ...]
        best_strategy_name = log_ensemble_results.pop("best_ensemble_strategy")
        if log_ensemble_results["best_ensemble_score"] > self.valid_best_ensemble_score:
            self.valid_best_ensemble_score = log_ensemble_results["best_ensemble_score"]
            self.valid_best_ensemble_strategy = best_strategy_name
        if not self.hparams.wandb_off:
            self.logger.experiment.log( {"valid_best_ensemble_strategy": best_strategy_name}) # equals to wandb.log()
        log_ensemble_results = dict(("valid_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
        log_dict.update(log_ensemble_results)

        self.log_dict(log_dict)
    
    def test_epoch_end(self, outputs):
        self.log_metrics('test')
        log_dict = dict()

        if self.hparams.layer == "all":
            if not isinstance(self.aggregator, nn.Conv1d):
                log_dict.update(self.log_weights('test'))
                

        log_ensemble_results = gs_ensemble(outputs, self.device) # [(strategy_name, predicted_label, label), ...]
        # additional handling the string log
        log_ensemble_results.pop("best_ensemble_strategy") # no need for test_best_strategy_name
        if self.valid_best_ensemble_strategy is not None:
            log_ensemble_results["ensemble_score_val-select"] = log_ensemble_results[f"ensemble_{self.valid_best_ensemble_strategy}_score"]
        # self.logger.experiment.config["test_best_ensemble_strategy"] = log_ensemble_results.pop("best_ensemble_strategy")
        log_ensemble_results = dict(("test_" + k, v) for (k,v) in log_ensemble_results.items()) # change the logging key name
        log_dict.update(log_ensemble_results)

        self.log_dict(log_dict)
    