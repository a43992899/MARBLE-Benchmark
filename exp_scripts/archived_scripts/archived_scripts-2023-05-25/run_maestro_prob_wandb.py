import json
import os
import sys

# running example, say there are 0-18 total 19 checkpoints will be loaded from the disk:
#  CUDA_VISIBLE_DEVICES=0 nohup python run_maestro_prob_wandb.py  0 6 > eval_transcription_0to6.log 2>&1 &
#  CUDA_VISIBLE_DEVICES=0 nohup python run_maestro_prob_wandb.py  6 10 > eval_transcription_6to10.log 2>&1 &
#  CUDA_VISIBLE_DEVICES=1 nohup python run_maestro_prob_wandb.py  10 15 > eval_transcription_10to15.log 2>&1 &
#  CUDA_VISIBLE_DEVICES=1 nohup python run_maestro_prob_wandb.py  15 19 > eval_transcription_15to19.log 2>&1 &

# specify which shards of the checkpoint list will be run.
s = int(sys.argv[1])
t = int(sys.argv[2])

ckpt_dir = '/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints/WaveEncoder'
ckpt_name_to_wandb_proj_json = '/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/selected_ckpt.json'
processed_dir = "/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/data/MAESTROv2/maestrov2_processed/hdf5s/maestro"

project_dir = '/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain'

with open(ckpt_name_to_wandb_proj_json) as json_file:
    data = json.load(json_file)
    # print(data['HF_HuBERT_base_MPD_train_1000h_valid_300h_iter1_250k_vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.5_ORI-0.1_ckpt_27_100000'])


# check existence
for k in data:
    abs_path = os.path.join(ckpt_dir, k)
    if not os.path.isdir(abs_path):
        print('could not found folder:', abs_path)

long_names = list(data.keys())[s:t]

print(f'running {s} to {t} checkpooints in total {len(data.keys())} checkpoints')

for k in long_names:
    wandb_project_name = data[k]
    exec_cmd = f"""
    cd {project_dir};
    python . probe \
        --dataset MAESTRO \
        --checkpoint {os.path.join(ckpt_dir, k)} \
        --feature_dir {processed_dir} \
        --metadata_dir NO \
        --num_outputs 88 \
        --monitor valid_loss \
        --earlystop_patience 30 \
        --lr_scheduler_patience 10 \
        --max_epochs 1 \
        --batch_size 12 \
        --lr 1e-3 \
        --layer all \
        --wandb_proj Eval_MAESTRO_{wandb_project_name} \
        --wandb_name layer_all_reduce_mean
    """
    print('excuting:', exec_cmd)
    os.system(exec_cmd)
