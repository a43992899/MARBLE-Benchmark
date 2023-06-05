#!/bin/bash

    {
    # step=250000
    # # # sample_rate=24000
    # model_list=( \
    #     MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v10_mix-v3-0.5_tau-0.1 \
    #     MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_tau-0.1 \
    #     MPD_1Kh_HPO-v4_crop5s_ECD-v12_mix-v3-0.5_tau-0.05 \
    #     MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.05 \
    #     MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.1 \
    # )

    # step=400000
    # # sample_rate=24000
    # model_list=( \
    #     MPD_1Kh_HPO-v4_crop10s_ECD-v10_mix-v3-0.5_tau-0.1)

    # step=250000
    # # sample_rate=16000
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
    # # sample_rate=16000
    # model_list=( \
    #     base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9 \
    #     MPD_1Kh_HPO-v4_crop5s_encodec-v4_mixture-v1-0.5)

    # step=95000
    steps=(55000 60000 65000 70000 75000 80000 85000)
    # model_list=(MuBERT_base_MPD-130Kh_HPO-v5_crop5s_grad-2)
    model_list=(MuBERT_large_MPD-130Kh-and-shenqi_HPO-v5_crop5s_grad-8-v5)
    # model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2)
    # model_list=(MuBERT_large_MPD-130Kh_HPO-v5_crop5s_grad-8-v2_extract_norm)
    # model_list=(MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5)


    # total_layer=14  # for base model 
    total_layer=26 # for large model

    # datasets="['GTZAN']"
    # datasets="['GS','GTZAN']"
    # datasets="['GTZAN','EMO']"
    datasets="['MTT','GTZAN','EMO']"
    # datasets="['GTZAN','GS','EMO']"
    # datasets="['MTT','GTZAN','GS','EMO']"
    # datasets="['MTT','GTZAN','GS','EMO','NSynthI']"
    # datasets="['MTT','GTZAN','GS','EMO','NSynthI','NSynthP']"
    # datasets="['NSynthI','NSynthP']"
    
    for i in "${!model_list[@]}"; do
        model_name=${model_list[i]}
        for j in "${!steps[@]}"; do
            step=${steps[j]}

            # log_name=eval.${model_name}.${step}
            # log_name=eval.GTZAN,GS,EMO,MTT,NSynthI,NSynthP.${model_name}.${step} # jukemir + nsynth
            # log_name=eval.GTZAN,GS,EMO,MTT.${model_name}.${step} # jukemir  四个任务
            log_name=eval.GTZAN,EMO,MTT.${model_name}.${step} # 补充 ensemble 任务
            # data/eval_log/eval.GTZAN,GS,EMO,MTT,NSynthI,NSynthP.MuBERT_base_MPD-130Kh_HPO-v5_crop5s_grad-2.50000.log

            python exp_scripts/filter_result_from_probing_log.py -tl ${total_layer} -l data/eval_log/${log_name}.log -s exp_results/ -o ${log_name} -d ${datasets}


            result_file="exp_results/${log_name}.json"
            python exp_scripts/filter_best_result_from_json.py -r ${result_file} -d ${datasets};
        done
    
    done

exit
}


