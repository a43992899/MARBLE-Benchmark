#!bin/bash

# copy from source dir to sync folder
# with proper renaming

# ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_1000h_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/*/*400000.pt
# for fp in `ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_1000h_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/*warmup*/*400000.pt`; do bash checkpoint_copy.sh ${fp}; done

# ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_10Kh_iter1/*/*400000.pt
# cd /share/project/music/music/MusicAudioPretrain_benchmark
# rlaunch --cpu=8 --gpu=1 --memory=8288 --private-machine=tenant --private-machine=group --charged-group=fujie -- bash
# dry-run print names:
# for ckpt in 240000 280000 320000 360000 400000 ; do  {  for fp in `ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_10Kh_iter1/*HPO-v4*/*${ckpt}.pt`; do echo ${fp}; done; } ; done
# for name in MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v2-0.25 MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v4-0.5  MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v6-0.25 \
# MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v6-0.5 MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v8-0.25; do { ls  /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/${name}/*_400000.pt; }; done
# actual run:
# for ckpt in 240000 280000 320000 360000 400000 ; do  {  for fp in `ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_10Kh_iter1/*HPO-v4*/*${ckpt}.pt`; do bash exp_scripts/baai_batch_convert_fairseq.sh ${fp}; done; } ; done
# for name in MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v2-0.25 MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v4-0.5  MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v6-0.25 \
# MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v6-0.5 MuBERT_MPD_1Kh_HPO-v4_crop5s_DC-v9_aug-v8-0.25; do { bash exp_scripts/baai_batch_convert_fairseq.sh  /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/${name}/*_400000.pt; }; done
# for name in MuBERT_MPD_1Kh_HPO-v4_crop5s_aug-v4-0.25_real ; do { bash exp_scripts/baai_batch_convert_fairseq.sh  /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/${name}/*_250000.pt; }; done

# ls /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/MuBERT_MPD_1Kh_HPO-v4_crop5s


CKPT_PATH=$1
# DEST_DIR=/share/project/music/music/checkpoint_sync

s=$(basename "${CKPT_PATH}")
run_name=$(dirname "${CKPT_PATH}")
run_name=$(basename ${run_name})

# echo ${run_name}_${step}
# echo $s
x="$(cut -d'_' -f2 <<<"$s")"_"$(cut -d'_' -f3 <<<"$s")" # xxxxxxxx.pt
step="$(cut -d'.' -f1 <<<"$x")"

echo ${run_name} and ${step}

# bash exp_scripts/convert_HuBERT_HF_script.sh local_test_baai music_hubert ${run_name} ${step} ${run_name} HuBERT_base_MPD_train_10Kh_iter1 config_musichubert
bash exp_scripts/convert_HuBERT_HF_script.sh local_test_baai music_hubert ${run_name} ${step} ${run_name} HuBERT_large_MPD_train_10Kh_iter1 config_musichubert_large
# [short]
# bash exp_scripts/baai_batch_convert_fairseq.sh /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/MuBERT_MPD_1Kh_HPO-v2_crop5s/checkpoint_162_100000.pt
# bash exp_scripts/baai_batch_convert_fairseq.sh /share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain/ckpt_HuBERT_base_MPD_train_1000h_iter1/MuBERT_MPD_1Kh_HPO-v2_crop5s/checkpoint_405_250000.pt


