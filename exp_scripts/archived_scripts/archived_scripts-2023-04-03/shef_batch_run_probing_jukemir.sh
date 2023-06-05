{

step=0
sample_rate=16000
model_list=(choi  chroma  clmr  jukebox  mfcc  musicnn)
# [(160,), (72,), (512,), (4800,), (120,), (4194,)]
model_setting_list=(1_160 1_72 1_512 1_4800 1_120 1_4194)

cuda_list=(0 1)
# export OMP_NUM_THREADS=2 # for baai

# model_list=(chroma  clmr)
# model_setting_list=(1_72 1_512)

# datasets="GS GTZAN MTT EMO"
# datasets="GS"

for i in "${!model_list[@]}"; do

    model_name=${model_list[i]}
    len_cuda=${#cuda_list[@]}
    cur_cuda_device=${cuda_list[$((i % len_cuda))]}
    
    model_setting=${model_setting_list[i]}

    echo "initailizing worker evaluate ${model_name} ${step} at cuda ${cur_cuda_device}, training sample rate ${sample_rate}"
    echo "model feature dimension ${model_setting}"

    CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        nohup bash exp_scripts/prob_jukemir_feature.sh jukemir local_test_shef \
        ${model_name} ${step} ${model_setting} ${sample_rate} > data/eval_log/eval.${model_name}.${step}.log 2>&1 &

    # sleep 1
done


exit
}