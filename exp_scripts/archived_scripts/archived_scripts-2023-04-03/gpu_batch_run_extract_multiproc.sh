{

step=60000; sample_rate=24000
# model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2)
model_list=(MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5)
PROCESSOR_NORMALIZE=False
MODEL_SETTING="24_1024"

cuda_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
export OMP_NUM_THREADS=4


datasets="GTZAN,GS,EMO,MTT,NSynth"
# datasets="NSynth,MTG"

# datasets="GTZAN,GS,EMO,NSynth,MTT,MTG"
# datasets="GTZAN,GS,EMO,NSynth"
# datasets="MTT,MTG"
ENABLE_EXTRACT=true
ENABLE_PROBE=false



N_SHARD=16
DEVICE_LIST='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
# DEVICE_LIST='8,9,10,11,12,13,14,15'



for i in "${!model_list[@]}"; do

    model_name=${model_list[i]}
    # len_cuda=${#cuda_list[@]}
    # cur_cuda_device=${cuda_list[$((i % len_cuda))]}
    
    echo "initailizing worker evaluate ${model_name} ${step} at cuda ${cur_cuda_device}, datasets ${datasets}, training sample rate ${sample_rate}"
    echo "ENABLE_EXTRACT: ${ENABLE_EXTRACT}, ENABLE_PROBE: ${ENABLE_PROBE}"

    log_name=data/eval_log/extract.${datasets}.${model_name}.${step}.sharded.log
    # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
    # MLU_VISIBLE_DEVICES=${cur_cuda_device} \
        nohup bash exp_scripts/prob_huggingface_checkpoint_multishard.sh hubert local_test_baai \
        ${model_name} ${step} ${MODEL_SETTING} ${sample_rate} ${datasets} ${ENABLE_EXTRACT} ${ENABLE_PROBE} ${N_SHARD} ${DEVICE_LIST} ${PROCESSOR_NORMALIZE} > ${log_name} 2>&1 &
    
    echo "see ${log_name}"
    sleep 1
done


exit
}