from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback

def get_callbacks(args):
    monitor_mode = 'min' if 'loss' in args.logger.monitor else 'max'
    checkpoint_naming = '{epoch}-{step}-{valid_loss:.4f}'
    if args.logger.monitor != 'valid_loss':
        checkpoint_naming += '-{' + args.logger.monitor + ':.4f}'
    checkpoint_callback = ModelCheckpoint(
        filename=checkpoint_naming, 
        monitor=args.logger.monitor,
        mode=monitor_mode, 
        save_top_k=1)
    
    early_stop_callback = EarlyStopping(
        args.logger.monitor, 
        patience=args.scheduler.earlystop_patience, 
        mode=monitor_mode, 
        verbose=True)
    return [checkpoint_callback, early_stop_callback]

