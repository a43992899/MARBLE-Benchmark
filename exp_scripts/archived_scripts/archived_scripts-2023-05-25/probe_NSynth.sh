#!/bin/bash

NUM_TRANS_LAYER=1

# FEATURE_DIR=data/NSynth/jukemir_features/jukemir_feature_jukemir_protocol
# NUM_FEATURES=4800

# FEATURE_DIR=data/NSynth/mule_features/mule_feature_jukemir_protocol
# NUM_FEATURES=1728

# FEATURE_DIR=data/NSynth/clmr_features/clmr_feature_jukemir_protocol
# NUM_FEATURES=512

FEATURE_DIR=data/NSynth/musicnn_features/musicnn_feature_jukemir_protocol
NUM_FEATURES=4194



ACCELERATOR=gpu
ACC_PRECISION=16
WANDB_OFF=false

LRs=(1e-4 5e-4 1e-3 5e-3 1e-2)

for LR in ${LRs[@]}
do

    for LAYER in -1 # all $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing NSynthI dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset NSynthI --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/NSynth/nsynth-data \
        --num_outputs 11 \
        --monitor valid_acc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
        --wandb_off ${WANDB_OFF}
    done

    for LAYER in -1 # $(seq ${NUM_TRANS_LAYER} -1 0)
    do
        echo "Probing NSynthP dataset with Layer: ${LAYER}"
        python -u deprecated__main__.py probe --dataset NSynthP --feature_dir ${FEATURE_DIR} \
        --metadata_dir data/NSynth/nsynth-data \
        --num_outputs 128 \
        --monitor valid_acc \
        --earlystop_patience 10 \
        --lr_scheduler_patience 3 \
        --max_epochs 100 \
        --hidden_layer_sizes '[512]' \
        --lr ${LR} \
        --layer ${LAYER} \
        --dropout_p 0.25 \
        --n_tranformer_layer ${NUM_TRANS_LAYER} --num_features ${NUM_FEATURES} \
        --accelerator ${ACCELERATOR} --precision ${ACC_PRECISION} \
        --wandb_off ${WANDB_OFF}
    done
done



