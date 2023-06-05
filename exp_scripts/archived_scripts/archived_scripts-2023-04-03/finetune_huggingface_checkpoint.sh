#!/bin/bash
{
# project_dir=/mnt/fastdata/acp21aao/MusicAudioPretrain_project_dir/MAP_benchmark
project_dir=/share/project/music/music/MAP_benchmark#
cd ${project_dir}


MODEL_SETTING=${1:-'12_768'}

# MODEL_TYPE=data2vec
MODEL_TYPE=hubert

OUTPUT_FEAT_ROOT=./data

# ckpt_dir=/home/yizhi/MusicAudioPretrain_project_dir/HF_checkpoints
ckpt_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts
# ckpt_dir=/share/project/music/music/hf_ckpt_short

# -------------- iterating checkpoing names --------------
# for name in HF_HuBERT_base_MPD_1Kh_LogMel-K-300-Chroma-K-200_m0.9-10_ckpt_67_250k HF_HuBERT_base_MPD_1Kh_Chroma-hier-ensemble-v1_ckpt_67_250k
# for name in HF_d2v_base_MPD_train_1000h_vanilla_d2v_vanila_cqt-pred-v9_ckpt_405_400k 
for name in HF_HuBERT_base_MPD_1Kh_HPO-baseline-v2_ckpt_134_250k

do 
    HF_CHECKPOINT_DIR=${ckpt_dir}/${name}
# ------------

# # ------------ iterating steps ---------------
# # for ckpt_step in 7_25000 14_50000 21_75000 34_125000
# for ckpt_step in  41_150000 47_175000 54_200000 61_225000
# do

#     HF_CHECKPOINT_DIR=${ckpt_dir}/HF_HuBERT_base_MPD_train_1000h_iter1_250k_vanilla_model_LogMel_229_1_300_Chroma_264_1_200_50Hz_ckpt_${ckpt_step::-3}k
# # ------------

    echo "evaluating checkpoint at ${HF_CHECKPOINT_DIR}"

    # # eval on EMO only
    bash exp_scripts/finetune_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} EMO ${MODEL_SETTING}

    # # eval on GS only
    # bash exp_scripts/finetune_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GS ${MODEL_SETTING}
    # # eval on GTZAN only
    # bash exp_scripts/finetune_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} GTZAN ${MODEL_SETTING}
    # # eval on MTT only
    # bash exp_scripts/finetune_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} MTT ${MODEL_SETTING}
done

exit
}
# CUDA_VISIBLE_DEVICES=0 nohup bash exp_scripts/finetune_huggingface_checkpoint.sh > eval.cuda0.GTZAN.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash exp_scripts/finetune_huggingface_checkpoint.sh > eval.cuda1.GS.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup bash exp_scripts/finetune_huggingface_checkpoint.sh > eval.cuda0.EMO.log 2>&1 &