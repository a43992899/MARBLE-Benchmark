


# sweep_id=bi2c994l ;dataset=GTZAN; agent_list=(0 1 2 3); cuda_list=(0 1 2 3)
# sweep_id=qrebk8cd ; dataset=GS; agent_list=(4 5 6 7); cuda_list=(4 5 6 7)
sweep_id=a2k6tc0x ;dataset=MTT; agent_list=(0 1 2 3 4 5 6 7); cuda_list=(8 9 10 11 12 13 14 15) # ; agent_list=(8 9 10 11)
# sweep_id=6q6uc2be  ; dataset=EMO; agent_list=(0 1 2 3); cuda_list=(0 1 2 3)

# agent_list=(0 1 2 3 4 5 6 7 8 9 10 11)

# agent_list=(0 1 2 3 4 5 6 7)
# cuda_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)



# model_name=jukebox
# model_name=MuBERT_MusicAll-900h-MPD-1Kh-fma-large_HPO-v4_crop5s_ECD-v8_mix-v3-0.5-tau-0.1
# model_name=MuBERT_large_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v13_mixture-v3-0.5 #_ckpt_28_200000_feature_layer_all_reduce_mean
model_name=MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_55_100000

export OMP_NUM_THREADS=2
# wandb agent musicaudiopretrain/sweep_MTT_probe_hidden_jukebox/40ek3fuq
for i in "${!agent_list[@]}"; do

    agent_id=${agent_list[i]}
    len_cuda=${#cuda_list[@]}
    cur_cuda_device=${cuda_list[$((i % len_cuda))]}

    log_name=data/eval_log/sweep_${sweep_id}.${dataset}_${model_name}.agent_${agent_id}.log
    echo "log to ${log_name}"
    # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
    MLU_VISIBLE_DEVICES=${cur_cuda_device} \
        nohup wandb agent musicaudiopretrain/sweep_${dataset}_probe_hidden/${sweep_id}  > ${log_name} 2>&1 &

done


# wandb sweep --project sweep_GTZAN_probe_hidden exp_scripts/sweep_config/sweep_GTZAN_probe_hidden_mlu.yaml
# wandb sweep --project sweep_GS_probe_hidden exp_scripts/sweep_config/sweep_GS_probe_hidden_mlu.yaml
# wandb sweep --project sweep_MTT_probe_hidden exp_scripts/sweep_config/sweep_MTT_probe_hidden_mlu.yaml
# wandb sweep --project sweep_EMO_probe_hidden exp_scripts/sweep_config/sweep_EMO_probe_hidden_mlu.yaml

# wandb sweep --project sweep_MTT_probe_hidden exp_scripts/sweep_config/sweep_MTT_probe_hidden.yaml
# wandb sweep --project sweep_GS_probe_hidden exp_scripts/sweep_config/sweep_GS_probe_hidden.yaml
# wandb sweep --project sweep_GTZAN_probe_hidden exp_scripts/sweep_config/sweep_GTZAN_probe_hidden.yaml
# wandb sweep --project sweep_EMO_probe_hidden exp_scripts/sweep_config/sweep_EMO_probe_hidden.yaml