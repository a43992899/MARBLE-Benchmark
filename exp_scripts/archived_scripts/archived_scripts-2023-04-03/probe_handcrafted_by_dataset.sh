#!/bin/bash
# Behavior: Probe a huggingface checkpoint's features on our benchmark.
# Path: exp_scripts/probe.sh
# Author: Ruibin Yuan
# Date: 2022-10-01
# Usage: conda active ${YOUR_ENV}
#        cd ${PROJECT_ROOT}
#        bash exp_scripts/probe.sh ${HF_CHECKPOINT_DIR}
# Note: Features should be extracted before probing (run exp_scripts/extract_hubert_features.sh).
{
HF_CHECKPOINT_DIR=$1
MODEL_TYPE=$2 # could be "hubert" or "data2vec"
OUTPUT_FEAT_ROOT=$3 # somewhere else or ./data

DATASET=$4

MODEL_NAME=${HF_CHECKPOINT_DIR##*/}
SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
MAX_NAME_LEN=120
# check if model name is too long
if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
    echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
    exit 1
fi

if [ "${HF_CHECKPOINT_DIR}" = "cqt" ]; then
    INPUT_FEATURE_DIM=2016
elif [ "${HF_CHECKPOINT_DIR}" = "chroma" ]; then
    INPUT_FEATURE_DIM=72
elif [ "${HF_CHECKPOINT_DIR}" = "mfcc" ]; then
    INPUT_FEATURE_DIM=120
fi

echo "Probing ${DATASET} dataset with model: ${MODEL_NAME}"
FEATURE_DIR=${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
# check if feature dir exists
if [ ! -d ${FEATURE_DIR} ]; then
    echo "Huggingface checkpoint dir does not exist"
    exit 1
fi
# bash exp_scripts/probe_${DATASET}.sh ${FEATURE_DIR}
if [ "${MODEL_TYPE}" = "handcrafted" ]; then
    echo "probing on handcrfated features, only one layer"
    LAYER=0
    if [ "${DATASET}" = "GTZAN" ];then
        echo "Probing GTZAN dataset with Layer: ${LAYER}"
        python . probe --dataset GTZAN --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GTZAN \
        --num_outputs 10 \
        --monitor valid_acc \
        --earlystop_patience 30 \
        --lr_scheduler_patience 10 \
        --max_epochs 400 \
        --lr 1e-2 \
        --layer ${LAYER} --num_features ${INPUT_FEATURE_DIM} \
        # --wandb_off true

    elif [ "${DATASET}" = "GS" ];then
        echo "Probing GS dataset with Layer: ${LAYER}"
        python . probe --dataset GS --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GS/giantsteps_clips \
        --num_outputs 24 \
        --monitor valid_best_ensemble_score \
        --earlystop_patience 30 \
        --lr_scheduler_patience 10 \
        --max_epochs 400 \
        --lr 1e-2 \
        --test_ensemble \
        --layer ${LAYER} --num_features ${INPUT_FEATURE_DIM} \
        # --wandb_off true

    elif [ "${DATASET}" = "MTT" ];then
        echo "Probing MTT dataset with Layer: ${LAYER}"
        python . probe --dataset MTT --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTT \
        --num_outputs 50 \
        --monitor valid_aucroc \
        --earlystop_patience 6 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --lr 1e-2 \
        --layer ${LAYER} --num_features ${INPUT_FEATURE_DIM} 
        # --wandb_off true
    fi
else
    echo "using the wrong script"
fi
exit
}