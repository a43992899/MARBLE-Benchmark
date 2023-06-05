#!/bin/bash
# Behavior: Extract features of a huggingface checkpoint and probe it on our benchmark.
# Path: exp_scripts/extract_and_probe.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/extract_and_probe.sh ${HF_CHECKPOINT_DIR}

HF_CHECKPOINT_DIR=$1
MODEL_TYPE=hubert # hubert of data2vec
OUTPUT_FEAT_ROOT=./data

# extract features
# bash exp_scripts/extract_hubert_features.sh ${HF_CHECKPOINT_DIR}
bash exp_scripts/extract_features.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT}

# probe 
bash exp_scripts/probe.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT}
