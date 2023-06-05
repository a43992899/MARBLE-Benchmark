import os
import wandb
from pytorch_lightning.loggers import WandbLogger

def get_pretrain_model_name(cfg):
    paradigm = cfg.trainer.paradigm
    if hasattr(cfg.dataset, 'pre_extract'):
        pretrain_model_name = cfg.dataset.pre_extract.feature_extractor.pretrain.name
    elif hasattr(cfg.model, 'feature_extractor'):
        pretrain_model_name = cfg.model.feature_extractor.pretrain.name
    else:
        raise NotImplementedError
    return pretrain_model_name

def get_logger(cfg):
    if cfg.logger.wandb_off: 
        return None
    
    paradigm = cfg.trainer.paradigm
    pretrain_model_name = get_pretrain_model_name(cfg)

    if cfg.logger.wandb_proj_name is None:
        if paradigm == 'probe':
            cfg.logger.wandb_proj_name = f"Probe_{pretrain_model_name}_on_{cfg.dataset.dataset}"
        elif paradigm == 'finetune':
            cfg.logger.wandb_proj_name = f"Finetune_{pretrain_model_name}_on_{cfg.dataset.dataset}"
        else:
            raise NotImplementedError 

    if cfg.logger.wandb_run_name is None:
        cfg.logger.wandb_run_name = pretrain_model_name
    
    if cfg.logger.wandb_sweep:
        cfg.logger.wandb_proj_name = f'Sweep_{cfg.logger.wandb_proj_name}'
        cfg.logger.wandb_run_name = f'Sweep_{cfg.logger.wandb_run_name}'

    wandb.init(
        project=cfg.logger.wandb_proj_name,
        entity='musicaudiopretrain', # team name
        name=cfg.logger.wandb_run_name, # run name
        dir=cfg.logger.wandb_dir
    )

    # save the argparser dictionary to wandb, this will be overwritten by wandb sweep
    wandb.config.update(cfg) 
    if cfg.logger.wandb_sweep:
        print('using sweep config to overwrite the argparse config')
        raise NotImplementedError("sweep is not supported yet")
        # TODO: wandb sweep is not supported yet
        #       need to support update cfg from wandb sweep config
        # for k in wandb.config.keys():
        #     setattr(cfg, k, wandb.config[k])

    logger = WandbLogger()

    return logger
