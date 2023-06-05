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
    MODEL_TYPE=$2 # i.e. hubert or data2vec
    OUTPUT_FEAT_ROOT=$3 # i.e. ./data
    TASK=$4
    MODEL_SETTING=${5:-'12_768'} # i.e. 12 layer transformer with 768 dim features

    ACCELERATOR=${6:-'mlu'} # mlu, gpu, cpu
    ACC_PRECISION=${7:-'32'} # force set 32 for mlu
    WANDB_OFF=${8:-'true'} # true, false


    NUM_TRANS_LAYER="$(cut -d'_' -f1 <<<"$MODEL_SETTING")"
    NUM_FEATURES="$(cut -d'_' -f2 <<<"$MODEL_SETTING")"
    echo "Transformer backbone has ${NUM_TRANS_LAYER} layers, featuren dimension ${NUM_FEATURES}"

    MODEL_NAME=${HF_CHECKPOINT_DIR##*/}
    SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
    # check if model name is too long
    MAX_NAME_LEN=120
    if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
        echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
        exit 1
    fi

    echo "Probing ${TASK} with model: ${MODEL_NAME}"

    if [ "${TASK}" = "GTZAN" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/GTZAN/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing GTZAN dataset with Layer: ${LAYER}"
            python -u . probe --dataset GTZAN --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/GTZAN \
            --num_outputs 10 \
            --monitor valid_acc \
            --earlystop_patience 20 \
            --lr_scheduler_patience 6 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 --num_workers 0 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --layer ${LAYER} --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "GS" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/GS/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing GS dataset with Layer: ${LAYER}"
            python -u . probe --dataset GS --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/GS/giantsteps_clips \
            --num_outputs 24 \
            --earlystop_patience 20 \
            --lr_scheduler_patience 5 \
            --max_epochs 400 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --test_ensemble \
            --layer ${LAYER} \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --monitor valid_best_ensemble_score --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "MTT" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/MTT/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing MTT dataset with Layer: ${LAYER}"
            python -u . probe --dataset MTT --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/MTT \
            --num_outputs 50 \
            --monitor valid_aucroc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 400 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --layer ${LAYER} --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "EMO" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/EMO/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing EMO dataset with Layer: ${LAYER}"
            python -u . probe --dataset EMO --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/EMO/emomusic \
            --num_outputs 2 \
            --monitor valid_r2 \
            --earlystop_patience 20 \
            --lr_scheduler_patience 5 \
            --max_epochs 1000 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 --num_workers 0 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "NSynthI" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/NSynth/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing NSynthI dataset with Layer: ${LAYER}"
            python -u . probe --dataset NSynthI --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/NSynth/nsynth-data \
            --num_outputs 11 \
            --monitor valid_acc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "NSynthP" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/NSynth/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing NSynthP dataset with Layer: ${LAYER}"
            python -u . probe --dataset NSynthP --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/NSynth/nsynth-data \
            --num_outputs 128 \
            --monitor valid_acc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "MTGMood" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing MTGMood dataset with Layer: ${LAYER}"
            python -u . probe --dataset MTGMood --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/MTG/mtg-jamendo-dataset \
            --num_outputs 56 \
            --monitor valid_aucroc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "MTGTop50" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing MTGTop50 dataset with Layer: ${LAYER}"
            python -u . probe --dataset MTGTop50 --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/MTG/mtg-jamendo-dataset \
            --num_outputs 50 \
            --monitor valid_aucroc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "MTGGenre" ];then
    {    
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0);  do
            echo "Probing MTGGenre dataset with Layer: ${LAYER}"
            python -u . probe --dataset MTGGenre --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/MTG/mtg-jamendo-dataset \
            --num_outputs 87 \
            --monitor valid_aucroc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "MTGInstrument" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/MTG/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing MTGInstrument dataset with Layer: ${LAYER}"
            python -u . probe --dataset MTGInstrument --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/MTG/mtg-jamendo-dataset \
            --num_outputs 40 \
            --monitor valid_aucroc \
            --earlystop_patience 10 \
            --lr_scheduler_patience 3 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "VocalSetT" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/VocalSet/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing VocalSetT dataset with Layer: ${LAYER}"
            python -u . probe --dataset VocalSetT --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/VocalSet \
            --num_outputs 10 \
            --num_workers 3 \
            --monitor valid_acc \
            --earlystop_patience 20 \
            --lr_scheduler_patience 5 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --layer ${LAYER} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
    elif [ "${TASK}" = "VocalSetS" ];then
    {
        FEATURE_DIR=${OUTPUT_FEAT_ROOT}/VocalSet/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing VocalSetS dataset with Layer: ${LAYER}"
            python -u . probe --dataset VocalSetS --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/VocalSet \
            --num_outputs 20 \
            --num_workers 3 \
            --monitor valid_acc \
            --earlystop_patience 20 \
            --lr_scheduler_patience 5 \
            --max_epochs 100 \
            --hidden_layer_sizes '[512]' \
            --lr 1e-3 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --layer ${LAYER} \
            --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
            --wandb_off ${WANDB_OFF}
        done
    }
fi
exit
}