#!/bin/bash

{
# FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000_feature_layer_all_reduce_mean
# NUM_TRANS_LAYER=12
# NUM_FEATURES=768

# FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2_ckpt_124_75000_feature_layer_all_reduce_mean
FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5_ckpt_60_55000_feature_layer_all_reduce_mean
CKPT_DIR=data/hf_ckpt/HF_MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5_ckpt_71_65000
NUM_TRANS_LAYER=24
NUM_FEATURES=1024

# ACCELERATOR=mlu
# ACC_PRECISION=32
ACCELERATOR=gpu
ACC_PRECISION=16

# for LAYER in all
for LAYER in 9
# for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        # echo "Probing GTZAN dataset with Layer: ${LAYER}"
        CUDA_VISIBLE_DEVICES=0 python -u . finetune --dataset GTZAN --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GTZAN --audio_dir data/GTZAN/genres \
        --pre_trained_folder ${CKPT_DIR} \
        --num_outputs 10 \
        --monitor valid_acc \
        --earlystop_patience 4 \
        --lr_scheduler_patience 2 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512,]' \
        --lr 1e-4 \
        --dropout_p 0.2 \
        --num_workers 8 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --layer ${LAYER} \
        --batch_size 4 \
        --wandb_off true \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
        --sliding_window_size_in_sec 5 --sliding_window_overlap_in_percent 0.0 \
        --train_sample_duration 30 --target_sr 24000 --processor_normalize false \
        --finetune_freeze_CNN 1 --finetune_freeze_bottom_transformer 18
        # --strategy ddp \
        # --devices 4 \
        # increase num_workers for chunking 
    done

# nohup bash exp_scripts/archived_scripts/finetune_GTZAN.sh > finetune.debug.log 2>&1 &
exit
}