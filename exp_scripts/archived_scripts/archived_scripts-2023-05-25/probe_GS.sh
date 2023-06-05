#!/bin/bash

# FEATURE_DIR=data/GS/mule_features/mule_protocol
# NUM_TRANS_LAYER=1
# NUM_FEATURES=1728

FEATURE_DIR=data/GS/hubert_features/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000_feature_layer_all_reduce_mean
NUM_TRANS_LAYER=12
NUM_FEATURES=768


# for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
# for LAYER in 0
for LAYER in 8 all
    do
        echo "Probing GS dataset with Layer: ${LAYER}"
        python -u . probe --dataset GS --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/GS/giantsteps_clips \
        --num_outputs 24 \
        --earlystop_patience 20 \
        --lr_scheduler_patience 5 \
        --max_epochs 400 \
        --hidden_layer_sizes '[512,]' \
        --lr 1e-4 \
        --test_ensemble \
        --layer ${LAYER} \
        --dropout_p 0.2 \
        --batch_size 64 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --wandb_off true \
        --monitor valid_best_ensemble_score
    done
