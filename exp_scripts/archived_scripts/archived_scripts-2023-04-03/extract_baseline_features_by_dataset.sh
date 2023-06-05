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

MODEL_TYPE=$1 # could be "clmr" "mule" "musicnn" "jukemir"

OUTPUT_FEAT_ROOT=$2 # /home/yizhi/map_features or data or handcrafted

DATASET=$3

N_SHARD=${4:-'1'}
DEVICE_LIST=${5:-'0,'}

ACCELERATOR=${6:-'gpu'} # mlu, gpu, cpu

echo "device list: ${DEVICE_LIST}"
DEVICE_LIST=${DEVICE_LIST//,/ }
DEVICE_LIST=(${DEVICE_LIST})
echo "total number of device: ${#DEVICE_LIST[@]}"

SUBFOLDER_NAME=${MODEL_TYPE}_feature_jukemir_protocol


if [ ! -d ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features ]; then
    echo "feature folder doesn't exist, create one:" ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features
    mkdir ${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features
fi

if [ "${DATASET}" = "GTZAN" ];then
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
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GTZAN/genres \
            --output_dir ${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR} &
        fi
    } done
fi

if [ "${DATASET}" = "GS" ];then
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
                --overwrite True \
                --n_shard ${N_SHARD} --shard_rank ${rank} \
                --accelerator ${ACCELERATOR}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/GS/giantsteps_clips/wav \
                --output_dir ${OUTPUT_FEAT_ROOT}/GS/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
                --overwrite True \
                --n_shard ${N_SHARD} --shard_rank ${rank} \
                --accelerator ${ACCELERATOR} &
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
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/MTT/mp3 \
            --output_dir ${OUTPUT_FEAT_ROOT}/MTT/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR} &
        fi
    } done
fi

if [ "${DATASET}" = "EMO" ];then
    echo extracting EMO features...
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # foreground process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/EMO/emomusic/wav \
            --output_dir ${OUTPUT_FEAT_ROOT}/EMO/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR}
        else
            # background process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/EMO/emomusic/wav \
            --output_dir ${OUTPUT_FEAT_ROOT}/EMO/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR} &
        fi
    } done
fi
if [ "${DATASET}" = "NSynth" ];then
    echo extracting NSynth features...
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # foreground process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/NSynth/nsynth-data \
            --output_dir ${OUTPUT_FEAT_ROOT}/NSynth/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR}
        else
            # background process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/NSynth/nsynth-data \
            --output_dir ${OUTPUT_FEAT_ROOT}/NSynth/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR} &
        fi
    } done
fi
if [ "${DATASET}" = "MTG" ];then
    echo extracting MTG features...
    echo "[Warning] Note that currently only support crop to first 30s, consider reimplementing using sliding window for full length audio as the others use."
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # foreground process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/MTG/audio-low \
            --output_dir ${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR}
        else
            # background process
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/MTG/audio-low \
            --output_dir ${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --shard_rank ${rank} \
            --n_shard ${N_SHARD} --accelerator ${ACCELERATOR} &
        fi
    } done
fi
if [ "${DATASET}" = "VocalSet" ];then
    for rank in $(seq 0 $(expr ${N_SHARD} - 1)); do {
        len_cuda=${#DEVICE_LIST[@]}
        cur_cuda_device=${DEVICE_LIST[$((rank % len_cuda))]}
        echo "extracting ${DATASET} features shard ${rank} on device ${cur_cuda_device}"
        if [ ${rank} = $(expr ${N_SHARD} - 1) ]; then
            # go_background=""
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/VocalSet/audio \
            --output_dir ${OUTPUT_FEAT_ROOT}/VocalSet/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR}
        else
            # go_background="&"
            CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            python . extract-${MODEL_TYPE}-features --audio_dir data/VocalSet/audio \
            --output_dir ${OUTPUT_FEAT_ROOT}/VocalSet/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
            --overwrite True \
            --n_shard ${N_SHARD} --shard_rank ${rank} \
            --accelerator ${ACCELERATOR} &
        fi
    } done
fi
exit
}