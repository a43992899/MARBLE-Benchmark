while true
do
    echo "waiting for 5 minutes to sync"
    sleep 5m
    rsync -avPz /sharefs/music/music/wandb/wandb   yizhi@67.171.69.161:/home/yizhi/baai_wandb
done
