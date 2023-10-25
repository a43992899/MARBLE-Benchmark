#!/bin/bash

# Function to display help/usage
display_help() {
    echo "Usage: $0 [OPTION]"
    echo
    echo "Choose the operation for the script to perform."
    echo
    echo "Options:"
    echo "  -h        Display this help message and exit."
    echo "  probe     Run the script in probe mode."
    echo "  extract   Run the script in extract mode."
    echo
    echo "Example:"
    echo "  $0 probe"
    echo "  $0 extract"
}

# Check if no arguments or -h is passed
if [ "$#" -eq 0 ] || [ "$1" == "-h" ]; then
    display_help
    exit 0
fi

OPERATION=$1

if [ "$OPERATION" != "probe" ] && [ "$OPERATION" != "extract" ]; then
    echo "Invalid choice. Use -h for help. Exiting..."
    exit 1
fi

if [ "$OPERATION" == "extract" ]; then
    # Perform operation on configs
    CONFIGS=(EMO GS GTZAN MTGGenre MTT NSynthI VocalSetS)

    for config in ${CONFIGS[@]}; do
        python . $OPERATION -c configs/mert/MERT-v1-330M/$config.yaml
    done
fi

if [ "$OPERATION" == "probe" ]; then
    TASKS=(GTZAN MTT EMO VocalSetT VocalSetS GS NSynthI NSynthP MTGGenre MTGInstrument MTGMood MTGTop50)
    lr_list=(1e-3 5e-3 5e-4 1e-2 1e-4 5e-5)

    for lr_value in ${lr_list[@]}; do
        for ((i=0; i<${#TASKS[@]}; i++)); do
            for LAYER in all $(seq 24 -1 0)
            do
                python . probe -c configs/mert/MERT-v1-330M/${TASKS[i]}.yaml \
                -o "optimizer.lr=${lr_value},,model.downstream_structure.components[0].layer='${LAYER}'"
            done
        done
    done
fi
