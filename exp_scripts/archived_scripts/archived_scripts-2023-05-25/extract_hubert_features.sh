#!/bin/bash
# Behavior: Extract huggingface hubert features on different datasets.
# Path: exp_scripts/extract_hubert_features.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/extract_hubert_features.sh ${HF_CHECKPOINT_DIR}
# Note: We extract and save the time-avg-pooled features of each layer, otherwise it takes too much space.

HF_CHECKPOINT_DIR=$1
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
    exit 1
fi

python . extract-hubert-features --audio_dir data/GTZAN/genres \
--output_dir data/GTZAN/hubert_features/${SUBFOLDER_NAME} \
--pre_trained_folder ${HF_CHECKPOINT_DIR} \
--overwrite True \
--reduction mean

python . extract-hubert-features --audio_dir data/GS/giantsteps_clips/wav \
--output_dir data/GS/hubert_features/${SUBFOLDER_NAME} \
--pre_trained_folder ${HF_CHECKPOINT_DIR} \
--overwrite True \
--reduction mean

python . extract-hubert-features --audio_dir data/MTT/mp3 \
--output_dir data/MTT/hubert_features/${SUBFOLDER_NAME} \
--pre_trained_folder ${HF_CHECKPOINT_DIR} \
--overwrite True \
--reduction mean
