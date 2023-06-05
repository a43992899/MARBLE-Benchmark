{
    # kill these probing pragrams and run on probing_2 machine
    model_list=( \
        large_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v13_mixture-v3-0.5 \
        # MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v10_mix-v3-0.5_tau-0.1 \
        # MPD-1Kh-MusicAll-900h-fma-large_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_tau-0.1 \
        # MPD_1Kh_HPO-v4_crop10s_ECD-v10_mix-v3-0.5_tau-0.1 \
        # MPD_1Kh_HPO-v4_crop5s_ECD-v12_mix-v3-0.5_tau-0.05 \
        # MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.05 \
        # MPD_1Kh_HPO-v4_crop5s_ECD-v8_mix-v1-0.5_tau-0.1 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9 \
        # base_MPD_train_1000h_valid_300h_iter1_HPO_LogMel-300-Chroma-200_crop-15s_m-prob-0.5-len-5_cqt-pred-1.0_baseline-v2 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_SGDR-v11 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-5 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v3-0.25 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v2-0.25_debug \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_cqt-m-2 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v1-0.25 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v1-0.25_DC-v8 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v8 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v11 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v10 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_SGDR-v7 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_aug-v1-0.25 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s_DC-v8 \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v2_crop5s \
        # base_MPD_train_10Kh_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s \
        # base_MPD_train_1000h_iter1_MuBERT_MPD_1Kh_HPO-v4_crop5s \
        )

    step=300000
    for i in "${!model_list[@]}"; do
        model_name=${model_list[i]}
        echo "killing ${model_name}"
        ps -aux | grep "${model_name} ${step}" | grep -v grep | awk '{print $2}' | xargs kill -9
        ps -aux | grep "${model_name}_ckpt" | grep -v grep | awk '{print $2}' | xargs kill -9
        # sleep 1

    done

exit
}