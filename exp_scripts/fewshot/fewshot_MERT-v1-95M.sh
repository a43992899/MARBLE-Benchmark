TASKS=(GTZAN GS VocalSetT VocalSetS)

for ((i=0; i<${#TASKS[@]}; i++)); do
    for LAYER in all $(seq 12 -1 0)
    do
        python . probe -c configs/mert/MERT-v1-95M/${TASKS[i]}.yaml \
        -o "model.downstream_structure.components[0].layer='${LAYER}'"
    done
done

