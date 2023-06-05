{

step=0
sample_rate=16000
model_list=( \
    base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9 \
    MPD_1Kh_HPO-v4_crop5s_encodec-v4_mixture-v1-0.5)


cuda_list=(0 1)
# export OMP_NUM_THREADS=2

# base
# MODEL_SETTING="12_768"
# large
# MODEL_SETTING="24_1024"

# jukemir
MODEL_SETTING="1_1024"

# step=100000
# step=400000


datasets="GS GTZAN MTT EMO"

for i in "${!model_list[@]}"; do

    model_name=${model_list[i]}
    len_cuda=${#cuda_list[@]}
    cur_cuda_device=${cuda_list[$((i % len_cuda))]}
    
    echo "initailizing worker evaluate ${model_name} ${step} at cuda ${cur_cuda_device}, datasets ${datasets}, training sample rate ${sample_rate}"

    CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        nohup bash exp_scripts/prob_huggingface_checkpoint.sh hubert local_test_shef \
        ${model_name} ${step} ${MODEL_SETTING} ${sample_rate} > data/eval_log/eval.${model_name}.${step}.log 2>&1 &

    sleep 1
done


exit
}