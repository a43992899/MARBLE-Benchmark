#!/bin/bash

# FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000_feature_layer_all_reduce_mean
# NUM_TRANS_LAYER=12
# NUM_FEATURES=768

# FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2_ckpt_124_75000_feature_layer_all_reduce_mean
# FEATURE_DIR=data/GTZAN/hubert_features/HF_MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5_ckpt_60_55000_feature_layer_all_reduce_mean
# NUM_TRANS_LAYER=24
# NUM_FEATURES=1024

ACCELERATOR=gpu
ACC_PRECISION=16

FEATURE_DIR=data/GTZAN/mule_features/mule_protocol
NUM_FEATURES=1728
NUM_TRANS_LAYER=1

# for LAYER in all
for LAYER in 0
# for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing GTZAN dataset with Layer: ${LAYER}"
        python -u . probe --dataset GTZAN --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GTZAN \
        --num_outputs 10 \
        --monitor valid_acc \
        --earlystop_patience 20 \
        --lr_scheduler_patience 5 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512,]' \
        --lr 1e-3 \
        --dropout_p 0.2 \
        --num_workers 0 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --layer ${LAYER} \
        --batch_size 64 \
        --wandb_off true \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION}
    done
