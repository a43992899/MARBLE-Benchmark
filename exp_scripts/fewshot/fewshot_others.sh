TASKS=(GTZAN GS VocalSetT VocalSetS)
MODELS=(clmr encodec jukemir mule musicnn)

for ((i=0; i<${#TASKS[@]}; i++)); do
    for ((j=0; j<${#MODELS[@]}; j++)); do
            python . fewshot -c configs/${MODELS[j]}/${TASKS[i]}.yaml
    done
done


