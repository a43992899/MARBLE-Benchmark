#!/bin/bash
# Behavior: Extract huggingface ${MODEL_TYPE} features on different datasets.
# Path: exp_scripts/extract_features.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/extract_features.sh ${FEATURE_TYPE} ${MODEL_TYPE}
# Note: We extract and save the time-avg-pooled features of each layer, otherwise it takes too much space.
{

FEATURE_TYPE=$1
MODEL_TYPE=$2 # could be "hubert" or "data2vec"

OUTPUT_FEAT_ROOT=$3 # /home/yizhi/map_features or data or handcrafted

DATASET=$4

NSHARD=4

MODEL_NAME=${FEATURE_TYPE##*/}
SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
MAX_NAME_LEN=120

# check if model name is too long
if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
    echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
    exit 1
fi
if [ "${MODEL_TYPE}" = "handcrafted" ]; then
        echo "Specifying extracting handcrafted feature ${FEATURE_TYPE}"
else
    echo "wrong input checkpoint dir"
    exit 1
fi


if [ "${DATASET}" = "GTZAN" ];then
    echo extracting GTZAN features
    python -u . extract-${MODEL_TYPE}-features --audio_dir data/GTZAN/genres \
        --output_dir ${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
        --feature_type ${FEATURE_TYPE} \
        --overwrite True \
        --reduction mean
fi

if [ "${DATASET}" = "GS" ];then
    echo extracting GS features
    for shard_rank in $(seq 0 $(expr ${NSHARD} - 1));do
    nohup    python . extract-${MODEL_TYPE}-features --audio_dir data/GS/giantsteps_clips/wav \
        --output_dir ${OUTPUT_FEAT_ROOT}/GS/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
        --feature_type ${FEATURE_TYPE} \
        --overwrite True \
        --reduction mean --nshard ${NSHARD} --shard_rank ${shard_rank} > extract.${DATASET}.${FEATURE_TYPE}.${shard_rank}.log 2>&1 &
    done
fi

if [ "${DATASET}" = "MTT" ];then
    echo extracting MTT features
    for shard_rank in $(seq 0 $(expr ${NSHARD} - 1));do
    nohup    python . extract-${MODEL_TYPE}-features --audio_dir data/MTT/mp3 \
    --output_dir ${OUTPUT_FEAT_ROOT}/MTT/${MODEL_TYPE}_features/${SUBFOLDER_NAME} \
    --feature_type ${FEATURE_TYPE} \
    --overwrite True \
    --reduction mean --nshard ${NSHARD} --shard_rank ${shard_rank} > extract.${DATASET}.${FEATURE_TYPE}.${shard_rank}.log 2>&1 &
    done
fi
exit
}