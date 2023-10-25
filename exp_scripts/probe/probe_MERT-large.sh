# python . extract -c configs/mert/MERT-v1-330M/EMO.yaml
# python . extract -c configs/mert/MERT-v1-330M/GS.yaml
# python . extract -c configs/mert/MERT-v1-330M/GTZAN.yaml
# python . extract -c configs/mert/MERT-v1-330M/MTGGenre.yaml
# python . extract -c configs/mert/MERT-v1-330M/MTT.yaml
# python . extract -c configs/mert/MERT-v1-330M/NSynthI.yaml
# python . extract -c configs/mert/MERT-v1-330M/VocalSetS.yaml


# TASKS=(GTZAN MTT EMO VocalSetT VocalSetS GS NSynthI NSynthP MTGGenre MTGInstrument MTGMood MTGTop50)
TASKS=(GTZAN)

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
