# python . extract -c configs/encodec/EMO.yaml
# python . extract -c configs/encodec/GS.yaml
# python . extract -c configs/encodec/GTZAN.yaml
# python . extract -c configs/encodec/MTGGenre.yaml
# python . extract -c configs/encodec/MTT.yaml
# python . extract -c configs/encodec/NSynthI.yaml
# python . extract -c configs/encodec/VocalSetS.yaml

# EMO.yaml  GS.yaml  GTZAN.yaml  MAESTRO.yaml  MTGGenre.yaml  MTGInstrument.yaml  MTGMood.yaml  MTGTop50.yaml  MTT.yaml  NSynthI.yaml  NSynthP.yaml  VocalSetS.yaml  VocalSetT.yaml

# TASKS=(GTZAN EMO GS VocalSetT VocalSetS MTT NSynthI NSynthP MTGGenre MTGInstrument MTGMood MTGTop50)

TASKS=(MTT)

lr_list=(1e-3 5e-3 5e-4 1e-2 1e-4 5e-5)

for lr_value in ${lr_list[@]}; do
    for ((i=0; i<${#TASKS[@]}; i++)); do
        # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        python . probe -c configs/encodec/${TASKS[i]}.yaml \
        -o "optimizer.lr=${lr_value}"
    done
done