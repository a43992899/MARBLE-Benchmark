#!/bin/bash

N_SHARD=1
DEVICE_LIST='0,'
ACCELERATOR='gpu'

# for MODEL in clmr musicnn mule jukemir
for MODEL in jukemir
do
    # for DATASET in VocalSet EMO GTZAN GS MTT NSynth MTG
    for DATASET in NSynth MTG
    do
        echo "Extracting ${MODEL} features for ${DATASET}"
        bash exp_scripts/extract_baseline_features_by_dataset.sh ${MODEL} data ${DATASET} ${N_SHARD} ${DEVICE_LIST} ${ACCELERATOR}
    done
done