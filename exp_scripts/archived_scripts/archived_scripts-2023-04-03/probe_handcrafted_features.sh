#!/bin/bash
{
project_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain
cd ${project_dir}

# python . extract-handcrafted-features --audio_dir data/GTZAN/genres \
# --output_dir data/GTZAN/Chroma_features/ \
# --pre_trained_folder 'Chroma'

# python . extract-handcrafted-features --audio_dir data/GTZAN/genres \
# --output_dir data/GTZAN/CQT_features/ \
# --pre_trained_folder 'CQT'

# python . extract-handcrafted-features --audio_dir data/GTZAN/genres \
# --output_dir data/GTZAN/MFCC_features/ \
# --pre_trained_folder 'MFCC'


# MODEL_TYPE=data2vec
# MODEL_TYPE=hubert
MODEL_TYPE=handcrafted

OUTPUT_FEAT_ROOT=./data

# ckpt_dir=/home/yizhi/MusicAudioPretrain_project_dir/HF_checkpoints
# ckpt_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts


for DATASET in MTT
#  GS GTZAN MTT
do 
{
    # for FEATURE_TYPE in mfcc chroma cqt
    for FEATURE_TYPE in cqt
    do 
    {
        echo "Extract on feature ${FEATURE_TYPE} for ${DATASET}"
        bash exp_scripts/extract_handcrafted_features_by_dataset.sh ${FEATURE_TYPE} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${DATASET}
        # echo "Probe on feature ${FEATURE_TYPE} for ${DATASET}"
        # bash exp_scripts/probe_handcrafted_by_dataset.sh ${FEATURE_TYPE} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${DATASET}
    }
    done
}
done

exit
}
# CUDA_VISIBLE_DEVICES=0 nohup bash exp_scripts/probe_handcrafted_features.sh > eval.cuda0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup bash exp_scripts/probe_handcrafted_features.sh > eval.cuda1.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup bash exp_scripts/probe_handcrafted_features.sh > extract_handcrafted.cuda1.log 2>&1 &