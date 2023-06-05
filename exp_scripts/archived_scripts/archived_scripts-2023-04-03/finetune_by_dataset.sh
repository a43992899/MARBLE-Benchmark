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
HF_CHECKPOINT_DIR=/home/yrb/data/hubert_data/HF_HuBERT_base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_ckpt_325_400000
MODEL_TYPE=hubert # could be "hubert" or "data2vec"
OUTPUT_FEAT_ROOT=/home/xingranc/MIR-Benchmark/checkpoints # somewhere else or ./data

DATASET=$4

MODEL_SETTING=${5:-'12_768'} # represent n-transormer layers and the dimension of final output

WANDB_OFF=${6:-'true'}

NUM_TRANS_LAYER="$(cut -d'_' -f1 <<<"$MODEL_SETTING")"
NUM_FEATURES="$(cut -d'_' -f2 <<<"$MODEL_SETTING")"

MODEL_NAME=${HF_CHECKPOINT_DIR##*/}
SUBFOLDER_NAME=${MODEL_NAME}_feature_layer_all_reduce_mean
MAX_NAME_LEN=120
# check if model name is too long
if [ ${#MODEL_NAME} -gt ${MAX_NAME_LEN} ]; then
    echo "Model name length should be less than or equal to ${MAX_NAME_LEN}, but got ${#MODEL_NAME}."
    exit 1
fi


echo "Fine-tuning ${DATASET} dataset with model: ${MODEL_NAME}"
FEATURE_DIR=${OUTPUT_FEAT_ROOT}/${DATASET}/${MODEL_TYPE}_features/${SUBFOLDER_NAME}
# # check if feature dir exists
# # if [ ! -d ${FEATURE_DIR} ]; then
# #     echo "Huggingface checkpoint dir does not exist"
# #     exit 1
# # fi
# bash exp_scripts/probe_${DATASET}.sh ${FEATURE_DIR}

if [ "${DATASET}" = "GTZAN" ];then
    for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Fine-tuning GTZAN dataset with Layer: ${LAYER}"
        python . finetune --dataset GTZAN --pre_trained_folder ${HF_CHECKPOINT_DIR} --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GTZAN \
        --audio_dir data/GTZAN/genres \
        --num_outputs 10 \
        --monitor valid_acc \
        --batch_size 8 \
        --accumulate_grad_batches 2 \
        --earlystop_patience 5 \
        --lr_scheduler_patience 1 \
        --max_epochs 400 \
        --lr 1e-4 \
        --layer ${LAYER} \
        --force_half true \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --train_sample_duration 15 --test_sample_duration 30 --wandb_off ${WANDB_OFF}
        # --test_ensemble False \
    done
elif [ "${DATASET}" = "GS" ];then
    for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Fine-tuning GS dataset with Layer: ${LAYER}"
        python . finetune --dataset GS --pre_trained_folder ${HF_CHECKPOINT_DIR} --feature_dir ${FEATURE_DIR}  \
        --metadata_dir data/GS/giantsteps_clips \
        --num_outputs 24 \
        --earlystop_patience 5 \
        --lr_scheduler_patience 1 \
        --max_epochs 400 \
        --force_half true \
        --lr 1e-4 \
        --batch_size 8 \
        --accumulate_grad_batches 2 \
        --train_sample_duration 15 \
        --test_sample_duration 30 \
        --layer ${LAYER} \
        --test_ensemble \
        --monitor valid_best_ensemble_score \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --audio_dir data/GS/giantsteps_clips/wav --wandb_off ${WANDB_OFF}
        # --eval_only --eval_ckpt_path data/GS/checkpoints/debug.ckpt \
        # --monitor valid_loss \
        # --eval_ckpt_path HPO-v2.GS_probing.layer_all.ckpt --eval_only
        # --best_ckpt_save HPO-v2.GS_probing.layer_${LAYER}.ckpt 

    done
elif [ "${DATASET}" = "MTT" ];then
    # do probe on MTT
    {   
        # for LAYER in all 
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Fine-tuning MTT dataset with Layer: ${LAYER}"
            python . finetune --dataset MTT --feature_dir ${FEATURE_DIR} \
            --metadata_dir "data/MTT" \
            --num_outputs 50 \
            --monitor valid_aucroc \
            --earlystop_patience 3 \
            --lr_scheduler_patience 1 \
            --max_epochs 400 \
            --force_half true \
            --lr 1e-4 \
            --batch_size 8 \
            --accumulate_grad_batches 2 \
            --train_sample_duration 15 \
            --test_sample_duration 30 \
            --layer ${LAYER} \
            --audio_dir "data/MTT/mp3" \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} --wandb_off ${WANDB_OFF}
            # --hidden_layer_sizes "[512,]" \
            # cc
    done
    }
elif [ "${DATASET}" = "EMO" ];then
    {   
        for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
        do
            echo "Probing EMO dataset with Layer: ${LAYER}"
            python . finetune --dataset EMO --feature_dir ${FEATURE_DIR} \
            --metadata_dir "data/EMO/emomusic" \
            --audio_dir "data/EMO/emomusic/wav" \
            --num_outputs 2 \
            --monitor valid_r2 \
            --earlystop_patience 5 \
            --lr_scheduler_patience 1 \
            --max_epochs 400 \
            --force_half true \
            --lr 1e-4 \
            --batch_size 8 \
            --accumulate_grad_batches 2 \
            --train_sample_duration 15 \
            --test_sample_duration 30 \
            --layer ${LAYER} \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} --wandb_off ${WANDB_OFF}
            # --hidden_layer_sizes "[512,]" \
            # cc
    done
    }
elif [ "${DATASET}" = "NSynth" ];then
    {   
        for LAYER in all 12 11 10 9 8 7 6 5 4 3 2 1 0
        do
            echo "Fine-tuning NSynth dataset with Layer: ${LAYER}"
            python . finetune --dataset EMO --feature_dir ${FEATURE_DIR} \
            --metadata_dir "data/EMO/emomusic" \
            --audio_dir "data/EMO/emomusic/wav" \
            --num_outputs 2 \
            --monitor valid_r2 \
            --earlystop_patience 5 \
            --lr_scheduler_patience 1 \
            --max_epochs 400 \
            --force_half true \
            --lr 1e-5 \
            --batch_size 8 \
            --accumulate_grad_batches 2 \
            --train_sample_duration 15 \
            --test_sample_duration 30 \
            --layer ${LAYER} \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} --wandb_off ${WANDB_OFF}
            # --hidden_layer_sizes "[512,]" \
            # cc
    done
    }
elif [ "${DATASET}" = "NSynth" ];then
    {   
        for LAYER in all 12 11 10 9 8 7 6 5 4 3 2 1 0
        do
            echo "Fine-tuning NSynth dataset with Layer: ${LAYER}"
            python . finetune --dataset EMO --feature_dir ${FEATURE_DIR} \
            --metadata_dir "data/EMO/emomusic" \
            --audio_dir "data/EMO/emomusic/wav" \
            --num_outputs 2 \
            --monitor valid_r2 \
            --earlystop_patience 5 \
            --lr_scheduler_patience 1 \
            --max_epochs 400 \
            --force_half true \
            --lr 1e-5 \
            --batch_size 8 \
            --accumulate_grad_batches 2 \
            --train_sample_duration 15 \
            --test_sample_duration 30 \
            --layer ${LAYER} \
            --pre_trained_folder ${HF_CHECKPOINT_DIR} --wandb_off ${WANDB_OFF}
            # --hidden_layer_sizes "[512,]" \
            # cc
    done
    }
else
    echo "NOT IMPLEMENTED"
    exit
fi

exit
}