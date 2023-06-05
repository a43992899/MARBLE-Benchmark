__conda_setup="$('/home/yrb/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" ;  eval "$__conda_setup"
conda activate map 

while true
do
    echo "waiting for 5 minutes to sync"
    sleep 5m
    wandb sync /home/yizhi/baai_wandb/wandb/run*
done
