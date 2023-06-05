#!/bin/bash
# Behavior: Probe a huggingface checkpoint's features on our benchmark.
# Path: exp_scripts/probe.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/probe.sh ${HF_CHECKPOINT_DIR}
# Note: Features should be extracted before probing (run exp_scripts/extract_hubert_features.sh).

HF_CHECKPOINT_DIR=$1
MODEL_TYPE=$2 # could be "hubert" or "data2vec"
OUTPUT_FEAT_ROOT=$3 # somewhere else or ./data


MODEL_NAME=${HF_CHECKPOINT_DIR##*/}
SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
MAX_NAME_LEN=120
# check if model name is too long
if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
    echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
    exit 1
fi
for DATASET in GS GTZAN MTT; do
# for DATASET in GS GTZAN; do
# for DATASET in MTT; do
    echo "Probing ${DATASET} dataset with model: ${MODEL_NAME}"
    FEATURE_DIR=${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
    # check if feature dir exists
    if [ ! -d ${FEATURE_DIR} ]; then
        echo "Huggingface checkpoint dir does not exist"
        exit 1
    fi
    bash exp_scripts/probe_${DATASET}.sh ${FEATURE_DIR}
done
