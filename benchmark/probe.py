import wandb
import argparse
import torch
import pytorch_lightning as pl

import benchmark as bench
from benchmark.utils.config_utils import load_config, override_config, print_config, merge_args_to_config
from benchmark.utils.get_dataloader import get_dataloaders
from benchmark.utils.get_logger import get_logger
from benchmark.utils.get_callbacks import get_callbacks 


def get_model(cfg):
    cfg._runtime.task_type = bench.TASK_TYPE_MAPPER[cfg.dataset.dataset]
    prober = eval(f"bench.{cfg.dataset.dataset}Prober")
    return prober(cfg)

def main(args):
    cfg = load_config(args.config, namespace=True)
    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, cfg)
    cfg = merge_args_to_config(args, cfg)
    print_config(cfg)

    cfg._runtime = argparse.Namespace() # runtime info

    assert cfg.trainer.paradigm == 'probe', "paradigm must be probe for probe.py"
    pl.seed_everything(cfg.trainer.seed)

    logger = get_logger(cfg)
    model = get_model(cfg)
    train_loader, valid_loader, test_loader = get_dataloaders(cfg)
    callbacks = get_callbacks(cfg)
    trainer = pl.Trainer.from_argparse_args(cfg.trainer, 
                                            logger=logger, 
                                            callbacks=callbacks, 
                                            default_root_dir="./data/lightning_logs",
                                            # amp_backend='apex',
                                            )
    if cfg.trainer.eval_only:
        eval_ckpt_path = cfg.checkpoint.eval_ckpt_path
        assert cfg.checkpoint.eval_ckpt_path, "must provide a checkpoint path for evaluation"
        assert cfg.trainer.strategy is None, "only support single device evaluation for now"
        trainer.validate(dataloaders=valid_loader, ckpt_path=eval_ckpt_path)
        trainer.test(dataloaders=test_loader, ckpt_path=eval_ckpt_path)
        return

    trainer.tune(model, train_loader, valid_loader)
    trainer.model.save_hyperparameters()
    trainer.fit(model, train_loader, valid_loader)
    if cfg.trainer.fast_dev_run: return

    # force single gpu test to avoid error
    strategy = cfg.trainer.strategy
    if strategy is not None: 
        assert "ddp" in strategy, "only support ddp strategy for now, other strategies may not get the right numbers"
        torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        # save best ckpt
        best_ckpt_path = trainer.checkpoint_callback.best_model_path

        cfg.trainer.devices = 1
        cfg.trainer.num_nodes = 1
        cfg.trainer.strategy = None
        
        trainer = pl.Trainer.from_argparse_args(cfg.trainer, 
                                            logger=logger
                                            )
        trainer.validate(model=model, dataloaders=valid_loader, ckpt_path=best_ckpt_path)
        trainer.test(model=model, dataloaders=test_loader, ckpt_path=best_ckpt_path)

        # does it really save the best model?
        if cfg.checkpoint.save_best_to is not None: trainer.save_checkpoint(cfg.checkpoint.save_best_to)

    wandb.finish()

