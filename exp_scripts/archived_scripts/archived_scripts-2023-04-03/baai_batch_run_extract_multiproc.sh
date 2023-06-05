{

################################### DONE #######################################
# step=250000
# sample_rate=16000
# model_list=( \
#     MERT_HPO-v1_crop-30s \
#     MERT_HPO-v1_crop-15s
# )
# step=300000
# sample_rate=24000
# model_list=(MuBERT_large_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v13_mixture-v3-0.5)

# model_list=(HuBERT_base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-5)

################################### RUNNING #######################################
# step=250000 ; sample_rate=24000
# #     MuBERT_MPD-1Kh-MusicAll-900h-fma-large-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_tau-0.1_nocqt
# model_list=(MuBERT_MPD1Kh-Musicall-fma-L-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_nocqt)
# model_list=(MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5)

# step=250000 ; sample_rate=16000
# model_list=(HPO-v1_crop-30s_mask-0.5-10)
# model_list=(HPO-v1_crop-30s_mask-0.65-10)
# model_list=(HPO-v1_crop-30s_mask-0.9-10)
# model_list=(HuBERT_base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-2)

################################### TODO #######################################


# steps=(250000); sample_rate=24000
# # model_list=(MuBERT_base_MPD-130Kh_HPO-v5_crop5s_grad-2)
# MODEL_SETTING="12_768"
# model_list=(MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5) # _3 在跑 extraction 

# ps aux | grep extract | grep -v grep | awk '{print $2}'| xargs kill
# cd /share/project/music/music/temp/MIR-Benchmark/ ; source ../../bench/bin/activate
# bash exp_scripts/baai_batch_run_extract_multiproc.sh

# step=65000; 
# steps=(90000 95000) # _2 在跑 extraction, 4个数据集都要跑
# steps=(55000 60000) # _3 在跑 extraction
# steps=(65000 70000 75000) # _5 在跑 extraction
# steps=(80000 85000) # _6 在跑 extraction 
steps=(100000) # _6 在跑 extraction 

sample_rate=24000
# model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2)
model_list=(MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5)
MODEL_SETTING="24_1024"

# cd /share/project/music/music/MIR-Benchmark ;  source /share/project/music/music/bench/bin/activate; bash exp_scripts/baai_batch_run_extract_multiproc.sh
PROCESSOR_NORMALIZE=False

# cuda_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
export OMP_NUM_THREADS=4


# datasets="GTZAN,GS,EMO,MTT,NSynth"
# datasets="NSynth,MTG"

# datasets="GTZAN,GS,EMO,NSynth,MTT,MTG"
# datasets="GTZAN,EMO,MTT"
datasets="GTZAN,GS,EMO,MTT"
# datasets="MTT,MTG"
ENABLE_EXTRACT=true
ENABLE_PROBE=false



N_SHARD=16
DEVICE_LIST='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
# DEVICE_LIST='8,9,10,11,12,13,14,15'



for i in "${!model_list[@]}"; do
    model_name=${model_list[i]}
    for j in "${!steps[@]}"; do
        
        step=${steps[j]}

        # len_cuda=${#cuda_list[@]}
        # cur_cuda_device=${cuda_list[$((i % len_cuda))]}
        
        echo "initailizing worker evaluate ${model_name} ${step} at cuda ${cur_cuda_device}, datasets ${datasets}, training sample rate ${sample_rate}"
        echo "ENABLE_EXTRACT: ${ENABLE_EXTRACT}, ENABLE_PROBE: ${ENABLE_PROBE}"

        log_name=data/eval_log/extract.${datasets}.${model_name}.${step}.sharded.log
        # CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        # MLU_VISIBLE_DEVICES=${cur_cuda_device} \
            nohup bash exp_scripts/prob_huggingface_checkpoint_multishard.sh hubert local_test_baai \
            ${model_name} ${step} ${MODEL_SETTING} ${sample_rate} ${datasets} \
            ${ENABLE_EXTRACT} ${ENABLE_PROBE} ${N_SHARD} ${DEVICE_LIST} ${PROCESSOR_NORMALIZE} > ${log_name} 2>&1 &
        
        echo "see ${log_name}"
        sleep 1
    done
done


exit
}