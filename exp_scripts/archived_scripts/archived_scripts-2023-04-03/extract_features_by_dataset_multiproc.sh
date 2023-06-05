#!/bin/bash
# Behavior: Extract huggingface ${MODEL_TYPE} features on different datasets.
# Path: exp_scripts/extract_features.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/extract_features.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE}
# Note: We extract and save the time-avg-pooled features of each layer, otherwise it takes too much space.
{

HF_CHECKPOINT_DIR=$1
MODEL_TYPE=$2 # could be "hubert" or "data2vec"

OUTPUT_FEAT_ROOT=$3 # /home/yizhi/map_features or data or handcrafted

DATASET=$4

TARGET_SAMPLE_RATE=${5:-'24000'}


N_SHARD=${6:-'4'}
DEVICE_LIST=${7:-'0,1,2,3,4,5,6,7'}

echo "device list: ${DEVICE_LIST}"
DEVICE_LIST=${DEVICE_LIST//,/ }
DEVICE_LIST=(${DEVICE_LIST})
echo "total number of device: ${#DEVICE_LIST[@]}"



MODEL_NAME=${HF_CHECKPOINT_DIR##*/}
SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
MAX_NAME_LEN=120
# check if model name is too long
if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
    echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
    exit 1
fi

# check if HF_CHECKPOINT_DIR exists
if [ ! -d ${HF_CHECKPOINT_DIR} ]; then
    echo "Huggingface checkpoint dir does not exist"
    if [ "${HF_CHECKPOINT_DIR}" = "random_model" ];then
        echo "Specifying using random huggingface model"
    # elif [ "${HF_CHECKPOINT_DIR}" = "CQT" ] || [ "${HF_CHECKPOINT_DIR}" = "Chroma" ] || [ "${HF_CHECKPOINT_DIR}" = "MFCC" ]; then
    elif [ "${MODEL_TYPE}" = "handcrafted" ]; then
         echo "Specifying extracting handcrafted feature ${HF_CHECKPOINT_DIR}"
    else
        echo "wrong input checkpoint dir"
        exit 1
    fi
fi

if [ ! -d ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features ]; then
    echo "feature folder doesn't exist, create one:" ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features
    mkdir ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features
fi

# for rank in $(seq 0 $(expr 10 - 1)); do echo $i; done

if [ "${DATASET}" = "GTZAN" ];then
    # echo extracting GTZAN features
    # python . extract-${MODEL_TYPE}-features --audio_dir data/GTZAN/genres \
    # --output_dir ${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
    # --target_sr ${TARGET_SAMPLE_RATE} \
    # --pre_trained_folder ${HF_CHECKPOINT_DIR} \
    # --overwrite True \
    # --reduction mean
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank} on device ${cur_cuda_device}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # go_background=""
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GTZAN/genres \
            --output_dir ${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GTZAN/genres \
            --output_dir ${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank} &
        fi
    } done
fi

if [ "${DATASET}" = "GS" ];then
    #  ${#DEVICE_LIST[@]}
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
                echo "extracting GS features shard ${rank} on device ${cur_cuda_device}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # go_background=""
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GS/giantsteps_clips/wav \
                --output_dir ${OUTPUT_FEAT_ROOT}/GS/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
                --target_sr ${TARGET_SAMPLE_RATE} \
                --pre_trained_folder ${HF_CHECKPOINT_DIR} \
                --overwrite True \
                --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GS/giantsteps_clips/wav \
                --output_dir ${OUTPUT_FEAT_ROOT}/GS/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
                --target_sr ${TARGET_SAMPLE_RATE} \
                --pre_trained_folder ${HF_CHECKPOINT_DIR} \
                --overwrite True \
                --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank} &
        fi
    } done
    # exit
fi

if [ "${DATASET}" = "MTT" ];then
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank} on device ${cur_cuda_device}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # go_background=""
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/MTT/mp3 \
            --output_dir ${OUTPUT_FEAT_ROOT}/MTT/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/MTT/mp3 \
            --output_dir ${OUTPUT_FEAT_ROOT}/MTT/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean --n_shard ${N_SHARD} --shard_rank ${rank} &
        fi
    } done
fi
if [ "${DATASET}" = "EMO" ];then
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank} on device ${cur_cuda_device}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # go_background=""
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/EMO/emomusic/wav \
            --output_dir ${OUTPUT_FEAT_ROOT}/EMO/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean \
            --crop_to_length_in_sec 30 \
            --crop_randomly False --n_shard ${N_SHARD} --shard_rank ${rank}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/EMO/emomusic/wav \
            --output_dir ${OUTPUT_FEAT_ROOT}/EMO/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --target_sr ${TARGET_SAMPLE_RATE} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} \
            --overwrite True \
            --reduction mean \
            --crop_to_length_in_sec 30 \
            --crop_randomly False --n_shard ${N_SHARD} --shard_rank ${rank} &
        fi
    } done
fi

# wait for the backgrounds
wait
echo "DONE extraction for ${HF_CHECKPOINT_DIR} on ${DATASET}" 
exit
}