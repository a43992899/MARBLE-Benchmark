python . extract -c configs/mert/MERT-v0/EMO.yaml
python . extract -c configs/mert/MERT-v0/GS.yaml
python . extract -c configs/mert/MERT-v0/GTZAN.yaml
python . extract -c configs/mert/MERT-v0/MTGGenre.yaml
python . extract -c configs/mert/MERT-v0/MTT.yaml
python . extract -c configs/mert/MERT-v0/NSynthI.yaml
python . extract -c configs/mert/MERT-v0/VocalSetS.yaml


TASKS=(GTZAN MTT EMO VocalSetT VocalSetS GS NSynthI NSynthP MTGGenre MTGInstrument MTGMood MTGTop50)
# DEVICE_LIST='1'
# DEVICE_LIST=${DEVICE_LIST//,/ }
# DEVICE_LIST=(${DEVICE_LIST})

lr_list=(1e-3 5e-3 5e-4 1e-2 1e-4 5e-5)

for lr_value in ${lr_list[@]}; do
    for ((i=0; i<${#TASKS[@]}; i++)); do
        for LAYER in all $(seq 12 -1 0)
        do
            # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
            python . probe -c configs/mert/MERT-v0/${TASKS[i]}.yaml \
            -o "optimizer.lr=${lr_value},,model.downstream_structure.components[0].layer='${LAYER}'"
            # "checkpoint.save_best_to=./best-layer-MERT-v1-95M/VocalSetT.ckpt"
        done
    done
done

