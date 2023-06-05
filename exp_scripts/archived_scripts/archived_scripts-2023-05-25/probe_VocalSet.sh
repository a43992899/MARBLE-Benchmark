#!/bin/bash

NUM_TRANS_LAYER=1

# FEATURE_DIR=data/VocalSet/jukemir_features/jukemir_feature_jukemir_protocol
# NUM_FEATURES=4800

# FEATURE_DIR=data/VocalSet/mule_features/mule_feature_jukemir_protocol
# NUM_FEATURES=1728

# FEATURE_DIR=data/VocalSet/clmr_features/clmr_feature_jukemir_protocol
# NUM_FEATURES=512

FEATURE_DIR=data/VocalSet/musicnn_features/musicnn_feature_jukemir_protocol
NUM_FEATURES=4194



ACCELERATOR=gpu
ACC_PRECISION=16
WANDB_OFF=false

LRs=(1e-4 5e-4 1e-3 5e-3 1e-2)

for LR in ${LRs[@]}
do

    for LAYER in -1 # all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing VocalSetT dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset VocalSetT --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/VocalSet \
        --num_outputs 10 \
        --num_workers 3 \
        --monitor valid_acc \
        --earlystop_patience 20 \
        --lr_scheduler_patience 5 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --layer ${LAYER} \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
        --wandb_off ${WANDB_OFF}
    done

    for LAYER in -1 # $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing VocalSetS dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset VocalSetS --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/VocalSet \
        --num_outputs 20 \
        --num_workers 3 \
        --monitor valid_acc \
        --earlystop_patience 20 \
        --lr_scheduler_patience 5 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --layer ${LAYER} \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
        --wandb_off ${WANDB_OFF}
    done
done

