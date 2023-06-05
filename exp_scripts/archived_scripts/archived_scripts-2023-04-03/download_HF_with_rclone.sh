
HF_store_dir=/home/yizhi/HF_checkpoints
cd ${HF_store_dir}

training_setting=HuBERT_base_MPD_train_1000h_valid_300h_iter1_250k
model_name=vanilla_model_Ensemble-2_LogMel-229-1_K-300_CQT-84-3_K-200
ckpt_step=27_100000

rclone lsd gd_shef:MusicAudioPretrain/MusicAudioPretrain_huggingface_checkpoints

echo HF_${training_setting}_${model_name}_ckpt_${ckpt_step}
rclone ls gd_shef:MusicAudioPretrain/MusicAudioPretrain_huggingface_checkpoints/HF_${training_setting}_${model_name}_ckpt_${ckpt_step}

mkdir HF_${training_setting}_${model_name}_ckpt_${ckpt_step}

rclone copy --ignore-checksum -P --drive-copy-shortcut-content \
gd_shef:MusicAudioPretrain/MusicAudioPretrain_huggingface_checkpoints/HF_${training_setting}_${model_name}_ckpt_${ckpt_step} \
./HF_${training_setting}_${model_name}_ckpt_${ckpt_step} 