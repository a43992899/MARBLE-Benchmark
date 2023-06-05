{


################################### base huggingface models #######################################
# step=250000
# sample_rate=24000
# model_list=( \
#     MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v10_mix-v3-0.5_tau-0.1 \
#     MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_tau-0.1 \
#     MPD_1Kh_HPO-v4_crop5s_ECD-v12_mix-v3-0.5_tau-0.05 \
#     MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.05 \
#     MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.1 \
#     MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_tau-0.1 \
# )

# step=400000
# sample_rate=24000
# model_list=( \
#     MPD_1Kh_HPO-v4_crop10s_ECD-v10_mix-v3-0.5_tau-0.1)

# step=250000
# sample_rate=16000
# model_list=( \
#     MusicAll-900h-MPD-1Kh_HPO-v4_crop5s_encodec-v5_mixture-v1-0.5 \
#     MusicAll-900h-MPD-1Kh_HPO-v4_crop5s_encodec-v1_mixture-v1-0.5 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v5_mixture-v1-0.5 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v6_mixture-v1-0.5 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v1_mixture-v1-0.5 \
#     MusicAll_900h_HPO-v4_crop5s_mixture-v1-0.25 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v3-0 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v3 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v2 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v1 \
#     base_MPD_train_1000h_iter1_MuBERT_MusicAll_900h_HPO-v4_crop5s \
#     MusicAll_900h_HPO-v4_crop5s \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_mixture-v2-0.5 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_mixture-v2-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_mixture-v1-0.5 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_mixture-v1-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v4-0.25_real \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v7-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v6-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v5-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v4-0.75 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v4-0.5 \
#     base_MPD_train_1000h_valid_300h_iter1_HPO_LogMel-300-Chroma-200_crop-15s_m-prob-0.5-len-5_cqt-pred-1.0_baseline-v2 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_SGDR-v11 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-5 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v3-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v2-0.25_debug \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-2 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v1-0.25 \
#     base_MPD_train_10Kh_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v1-0.25_DC-v8 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v8 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v11 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v10 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v7 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_aug-v1-0.25 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_DC-v8 \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s \
#     )

# step=400000
# sample_rate=16000
# model_list=( \
#     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9 \
#     MPD_1Kh_HPO-v4_crop5s_encodec-v4_mixture-v1-0.5)


################################### large huggingface models #######################################
# step=100000
# step=200000
# step=400000
# sample_rate=24000
# model_list=( \
#     large_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v13_mixture-v3-0.5 \
# )

# step=200000
# sample_rate=16000
# model_list=( \
#     large_MPD_10Kh_HPO-v4_crop5s_DC-v9 \
# )

# step=300000
# step=600000
# sample_rate=16000
# model_list=( \
#     large_MPD_train_10Kh_iter1_MuBERT_large_MPD_10Kh_HPO-v4_crop5s_DC-v9 \
# )

# step=250000; sample_rate=16000
# model_list=( \
#     MERT_HPO-v1_crop-30s \
#     MERT_HPO-v1_crop-15s
# )
# step=250000; sample_rate=24000
# model_list=( \
#     MuBERT_MPD1Kh-Musicall-fma-L-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_nocqt \
# )
# model_list=( \
#     MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5
# )

# model_list=(MuBERT_base_MPD-130Kh-and-others_HPO-v5_crop5s_250K-grad-2)
# model_list=(MuBERT_base_MPD-130Kh-and-others_HPO-v5_crop5s)
# model_list=(MuBERT_base_YTB-130Kh-and-others_HPO-v5_crop5s)
# model_list=(MuBERT_base_MPD-130Kh-and-others_HPO-v5_crop5s_mel-pred-m-1)

#### RUNNING ####
# step=75000; sample_rate=24000
# model_list=(MuBERT_base_shenqi-v2_HPO-v5_crop5s_grad-2_con-150K-YTB-gard-2-v1)

#### TODO ####

# ps aux | grep prob | grep -v grep | awk '{print $2}'| xargs kill
# cd /share/project/music/music/temp/MIR-Benchmark/ ; source ../../bench/bin/activate
# bash exp_scripts/baai_batch_run_probing.sh

# step=150000; 
# sample_rate=24000
# steps=(250000)
# # model_list=(MuBERT_base_MPD-130Kh_HPO-v5_crop5s_grad-2)
# model_list=(MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5)
# MODEL_SETTING="12_768"


steps=(100000) #
# steps=(90000 95000) # 
# steps=(55000 60000 65000 70000 75000 80000 85000) # 只需要补充 GTZAN EMO MTT 数据集 做 ensemble

# step=65000; 
sample_rate=24000
# model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2)
# model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2_extract_norm)
model_list=(MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5)
MODEL_SETTING="24_1024"

cuda_list=(11 12 13 14 15 0 1 2 3 4 5 6 7 8 9 10)
export OMP_NUM_THREADS=4

# datasets="GTZAN,GS,EMO,MTT,NSynthI,NSynthP"
# datasets="NSynthI,NSynthP"
# datasets="NSynthI,NSynthP,MTGMood,MTGTop50,MTGGenre,MTGInstrument"

# datasets="GTZAN,GS,EMO,MTT,NSynthI,NSynthP,MTGMood,MTGTop50,MTGGenre,MTGInstrument"
# datasets="MTT"

datasets="GTZAN,GS,EMO,MTT" # jukemir
# datasets="GTZAN,EMO,MTT" # probing with ensembled protocol

ENABLE_EXTRACT=false
ENABLE_PROBE=true

export WANDB_MODE="offline"

for i in "${!model_list[@]}"; do

    model_name=${model_list[i]}
    len_cuda=${#cuda_list[@]}
    len_step=${#steps[@]}
    
    for j in "${!steps[@]}"; do
        step=${steps[j]}
        cur_cuda_device=${cuda_list[$(((i * len_step + j) % len_cuda))]}
        
        echo "initailizing worker evaluate ${model_name} ${step} at cuda ${cur_cuda_device}, datasets ${datasets}, training sample rate ${sample_rate}"
        echo "ENABLE_EXTRACT: ${ENABLE_EXTRACT}, ENABLE_PROBE: ${ENABLE_PROBE}"

        # log_name=data/eval_log/eval.${datasets}.${model_name}.${step}.fixed.log
        log_name=data/eval_log/eval.${datasets}.${model_name}.${step}.log

        CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
        MLU_VISIBLE_DEVICES=${cur_cuda_device} \
        nohup bash exp_scripts/prob_huggingface_checkpoint_multishard.sh hubert local_test_baai \
            ${model_name} ${step} ${MODEL_SETTING} ${sample_rate} ${datasets} ${ENABLE_EXTRACT} ${ENABLE_PROBE} \
            > ${log_name} 2>&1 &
        echo "see ${log_name}"
        sleep 1

    done
done


exit
}