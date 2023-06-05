#!/bin/bash
# example: bash exp_scripts/prob_huggingface_checkpoint.sh hubert
{
    MODEL_TYPE=${1:-'jukemir'} # hubert, data2vec
    ENV_SETTING=${2:-'local_test_baai'}
    MODEL_NAME=$3
    CKPT_STEP=$4
    MODEL_SETTING=${5:-'12_768'}
    TARGET_SAMPLE_RATE=${6:-'16000'}
    IS_EXTRACT=${7:-'true'}

    case $ENV_SETTING in
        local_test_default)
            project_dir=./
            ckpt_dir=./data/hubert_data
            ;;
        local_test_shef)
            project_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR-Benchmark
            ckpt_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts
            ;;
        local_test_baai)
            # project_dir=/share/project/music/music/MAP_benchmark
            # ckpt_dir=/share/project/music/music/hf_ckpt_short
            project_dir=./
            ckpt_dir=./data/hubert_data
            
            # export WANDB_API_KEY=bf72b4f3341b5e72073c1ad84856802f67703ed6
            export WANDB_DIR=/share/project/music/music/wandb
            export WANDB_MODE=offline
            # export WANDB_MODE=online
            export WANDB_RESUME=allow
            export HYDRA_FULL_ERROR=1
            ;;
        *)
            echo "Unknown setting: $ENV_SETTING"
            exit 1
            ;;
    esac

cd ${project_dir}
echo ${project_dir}
# exit 
OUTPUT_FEAT_ROOT=./data


# ckpt_dir=/home/yizhi/MusicAudioPretrain_project_dir/HF_checkpoints

# ckpt_dir=/home/yizhi/HF_short_ckpt
# for ckpt in 13_240k 15_280k 20_360k 18_320k; do
# for ckpt in 39_240k 45_280k 52_320k 58_360k; do


# for HF_CHECKPOINT_DIR in `ls -d ${ckpt_dir}/*_${MODEL_NAME}_ckpt_*_${CKPT_STEP}`;
# do 
    # HF_CHECKPOINT_DIR=${ckpt_dir}/${name}


    for dataset in MTT GTZAN GS EMO; do {
    # for dataset in GS GTZAN; do {
    # for dataset in MTT GTZAN; do {
    # for dataset in GS EMO; do {
        echo "evaluating checkpoint at ${MODEL_NAME} for dataset ${dataset}, model type ${MODEL_TYPE}, audio sample rate ${TARGET_SAMPLE_RATE}"
        # if [ "${IS_EXTRACT}" = "true" ];then
        #     bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${dataset} ${TARGET_SAMPLE_RATE}
        # fi
        bash exp_scripts/probe_by_dataset.sh ${MODEL_NAME} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${dataset} ${MODEL_SETTING}
    }
    done

    

# done
# done
exit
}