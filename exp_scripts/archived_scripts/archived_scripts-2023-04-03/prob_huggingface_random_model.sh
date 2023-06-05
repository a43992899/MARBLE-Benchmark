#!/bin/bash
{
project_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain
cd ${project_dir}



# MODEL_TYPE=data2vec
MODEL_TYPE=hubert

OUTPUT_FEAT_ROOT=./data

# ckpt_dir=/home/yizhi/MusicAudioPretrain_project_dir/HF_checkpoints
ckpt_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts

# for name in HF_d2v_base_MPD_train_1000h_vanilla_d2v_vanila_cqt-pred-v9_ckpt_81_80k
for name in HF_HuBERT_base_MPD_train_1000h_valid_300h_iter1_vanilla_jukebox_5b_K-2048_v1_ckpt_15_50k
do 
    
    # HF_CHECKPOINT_DIR=${ckpt_dir}/${name}
    HF_CHECKPOINT_DIR="random_model"

    echo "evaluating checkpoint at ${HF_CHECKPOINT_DIR}"

    # eval on GS only
    bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GS
    bash exp_scripts/probe_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GS
    # eval on GTZAN only
    bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GTZAN
    bash exp_scripts/probe_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GTZAN
    # eval on MTT only
    bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} MTT
    bash exp_scripts/probe_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} MTT
done

exit
}
# CUDA_VISIBLE_DEVICES=0 nohup bash exp_scripts/prob_huggingface_random_model.sh > eval.cuda0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash exp_scripts/prob_huggingface_random_model.sh > eval.cuda1.log 2>&1 &