mkdir -p data

ln -sn /share/project/music/music/mir_benchmark_data/EMO ./data/EMO
ln -sn /share/project/music/music/hf_ckpt/WaveEncoder ./data/hubert_data
ln -sn /share/project/music/music/mir_benchmark_data/GTZAN ./data/GTZAN
ln -sn /share/project/music/music/mir_benchmark_data/MTT ./data/MTT
ln -sn /share/project/music/music/mir_benchmark_data/GS ./data/GS
# ln -sn /home/yrb/data/MAESTRO ./data/MAESTRO
# ln -sn /home/yrb/data/spotify_million_playlist/MPD ./data/MPD
ln -sn /share/project/music/music/wandb ./data/wandb
ln -sn /share/project/music/music/mir_benchmark_data/eval_log ./data/eval_log
