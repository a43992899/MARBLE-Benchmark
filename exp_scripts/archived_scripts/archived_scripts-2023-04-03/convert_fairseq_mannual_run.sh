
{
cd /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_target-12 227_240000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_music d2v_vanila_cqt-pred-v10 227_240000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_crop10s 1240_400000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_crop15s 808_400000

# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_prob50 378_400000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_prob70 378_400000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_prob80 378_400000

# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_span5 378_400000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_span15 378_400000

# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_crop5s 2534_400000

# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-100-Chroma-400_400k 216_400000 HPO-v2-based_L100C400_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-200-Chroma-300_400k 216_400000 HPO-v2-based_L200C300_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_400k 216_400000 HPO-v2-based_L300C200_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-400-Chroma-100_400k 216_400000 HPO-v2-based_L400C100_400k

# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-500_400k 216_400000 HPO-v2-based_L500_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_Chroma-500_400k 216_400000 HPO-v2-based_C500_400k

# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-16k_400k 216_400000 HPO-v2-based_L300C200_warmup-16k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-48k_400k 216_400000 HPO-v2-based_L300C200_warmup-48k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-64k_400k 216_400000 HPO-v2-based_L300C200_warmup-64k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-80k_400k 216_400000 HPO-v2-based_L300C200_warmup-80k_400k

# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-16k_400k 216_400000 HPO-v2-based_L300C200_warmup-16k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-48k_400k 216_400000 HPO-v2-based_L300C200_warmup-48k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-64k_400k 216_400000 HPO-v2-based_L300C200_warmup-64k_400k
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-300-Chroma-200_warmup-80k_400k 216_400000 HPO-v2-based_L300C200_warmup-80k_400k

# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3 22_400000 MuBERT_MPD_10Kh_HPO-v3
# TODO: cqt0 的接口似乎又问题， 在HF model里面好像强行设了 512+336，得加上判断
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_cqt0 22_400000 MuBERT_MPD_10Kh_HPO-v3_cqt0
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_aug-v1-0.25 22_400000 MuBERT_MPD_10Kh_HPO-v3_aug-v1-0.25
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_aug-v1-0.5 22_400000 MuBERT_MPD_10Kh_HPO-v3_aug-v1-0.5
# bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_crop5s 65_400000 MuBERT_MPD_10Kh_HPO-v3_crop5s
# TODO: 为什么这个也是 22 epoch？ 没设对？config似乎没错
# # bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_crop10s 22_400000 MuBERT_MPD_10Kh_HPO-v3_crop10s
# for fp in `ls /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_ckpt_baai/checkpoint_sync/*.pt `;do
# {
#     # run_name=$(dirname "${fp}")
#     # run_name=$(basename ${run_name})
#     run_name=$(basename "${fp}")

#     s=$(basename "${fp}")

#     x="$(cut -d'_' -f2 <<<"$s")"_"$(cut -d'_' -f3 <<<"$s")" # xxxxxxxx.pt
#     step="$(cut -d'.' -f1 <<<"$x")"

#     echo ${run_name}
#     echo ${step}
# }
# done

for ckpt in 13_240000 15_280000 18_320000 20_360000 22_400000;do
    bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_crop10s ${ckpt}
done

for ckpt in 17_240000 19_280000 22_320000 24_360000 26_400000;do
    bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_DC-v4 ${ckpt}
done

for ckpt in 17_240000 18_280000 20_320000 21_360000 22_400000;do
    bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_DC-v5 ${ckpt}
done

for ckpt in 17_240000 19_280000 21_320000 23_360000 25_400000;do
    bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_DC-v6 ${ckpt}
done

for ckpt in 19_240000 21_280000 24_320000 26_360000 28_400000;do
    bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_DC-v7 ${ckpt}
done

exit
}
# for ckpt in 39_240000 45_280000 52_320000 58_360000; do 
#     bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert MuBERT_MPD_10Kh_HPO-v3_crop5s ${ckpt} MuBERT_MPD_10Kh_HPO-v3_crop5s
# done

