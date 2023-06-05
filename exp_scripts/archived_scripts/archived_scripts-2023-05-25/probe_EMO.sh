#!/bin/bash
FEATURE_DIR=data/EMO/MERT_features/MERT-v1-330M_feature_default
NUM_TRANS_LAYER=24
NUM_FEATURES=1024

ACCELERATOR=gpu
ACC_PRECISION=16

LRs=(1e-3 5e-4 5e-3 1e-4 1e-2)

for LR in ${LRs[@]}
do
    for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
    # for LAYER in 0
        do
            echo "Probing EMO dataset with Layer: ${LAYER}"
            python -u deprecated__main__.py probe --dataset EMO --feature_dir ${FEATURE_DIR} \
            --metadata_dir data/EMO/emomusic \
            --num_outputs 2 \
            --monitor valid_r2 \
            --earlystop_patience 20 \
            --lr_scheduler_patience 5 \
            --max_epochs 1000 \
            --hidden_layer_sizes '[512]' \
            --lr ${LR} --num_workers 0 \
            --layer ${LAYER} \
            --dropout_p 0.25 \
            --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES}
        done
done
