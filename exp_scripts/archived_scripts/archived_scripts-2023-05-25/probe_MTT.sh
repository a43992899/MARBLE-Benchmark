#!/bin/bash

FEATURE_DIR=data/MTT/hubert_features/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000_feature_layer_all_reduce_mean
NUM_TRANS_LAYER=12
NUM_FEATURES=768

for LAYER in all # $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing MTT dataset with Layer: ${LAYER}"
        python __main__.py probe --dataset MTT --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTT \
        --num_outputs 50 \
        --monitor valid_aucroc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 400 \
        --hidden_layer_sizes '[512]' \
        --lr 1e-3 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --layer ${LAYER} \
        --devices 1 \
        --precision 16 \
        --accelerator gpu \
        --wandb_off true \
        --auto_lr_find false 
        # --strategy ddp
    done
