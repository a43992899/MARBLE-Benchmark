#!/bin/bash

NUM_TRANS_LAYER=1

# FEATURE_DIR=data/MTG/jukemir_features/jukemir_feature_default
# NUM_FEATURES=4800

# FEATURE_DIR=data/MTG/mule_features/mule_feature_default
# NUM_FEATURES=1728

# FEATURE_DIR=data/MTG/musicnn_features/musicnn_feature_default
# NUM_FEATURES=4194

FEATURE_DIR=data/MTG/clmr_features/clmr_feature_default
NUM_FEATURES=512

LRs=(1e-3 5e-4 5e-3 1e-4 1e-2)

for LR in ${LRs[@]}
do

    for LAYER in -1 # all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing MTGInstrument dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset MTGInstrument --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTG/mtg-jamendo-dataset \
        --num_outputs 40 \
        --monitor valid_aucroc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --num_workers 8 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES}
    done

    for LAYER in -1 # $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing MTGGenre dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset MTGGenre --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTG/mtg-jamendo-dataset \
        --num_outputs 87 \
        --monitor valid_aucroc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES}
    done

    for LAYER in -1 # all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing MTGMood dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset MTGMood --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTG/mtg-jamendo-dataset \
        --num_outputs 56 \
        --monitor valid_aucroc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES}
    done

    for LAYER in -1 # all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing MTGTop50 dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset MTGTop50 --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/MTG/mtg-jamendo-dataset \
        --num_outputs 50 \
        --monitor valid_aucroc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES}
    done
done