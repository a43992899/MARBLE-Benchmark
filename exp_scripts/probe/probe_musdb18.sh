# NUM_TRANS_LAYER=12
# NUM_FEATURES=768
# PRETRAIN_FOLDER="data/hubert_data/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000"

NUM_TRANS_LAYER=12
NUM_FEATURES=768
PRETRAIN_FOLDER="/home/music/data/hubert_data/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000"

for LAYER in all $(seq ${NUM_TRANS_LAYER} -1 0)
# for LAYER in 7 6 5 4 3 2 1 0
# for LAYER in 7
    do
        echo "Probing MUSDB18 dataset with Layer: ${LAYER}"
        python __main__.py probe \
        --dataset MUSDB18 \
        --pre_trained_model_type hubert \
        --pre_trained_folder ${PRETRAIN_FOLDER} \
        --target_sr 24000 \
        --token_rate 75 \
        --audio_dir /home/music/data/MUSDB18 \
        --metadata_dir /home/music/data/MUSDB18 \
        --num_outputs 88 \
        --monitor valid_sdr \
        --earlystop_patience 5 \
        --lr_scheduler_patience 2 \
        --max_epochs 50 \
        --batch_size 32 \
        --accumulate_grad_batches 1 \
        --token_rate 75 \
        --frame_threshold 0.5 \
        --lr 1e-3 \
        --layer ${LAYER} \
        --num_features ${NUM_FEATURES} \
        --n_tranformer_layer ${NUM_TRANS_LAYER} \
        --hidden_layer_sizes '[512]' \
        --target 'vocals.wav' \
        --num_workers 8 \
        --debug False \
        --strategy ddp \
        --devices 8 \
        --auto_lr_find True \
        --wandb_off True \
        # --strategy ddp_find_unused_parameters_false \
    done