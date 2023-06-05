NUM_TRANS_LAYER=12
NUM_FEATURES=768

cuda_list=(0 1 2 3 4 5 6 7)
# all $(seq ${NUM_TRANS_LAYER} -1 0)
layer_list=(all 12 11 10 9 8 7 6 5 4 3 2 1 0)

for i in "${!layer_list[@]}"; do

    len_cuda=${#cuda_list[@]}
    cur_cuda_device=${cuda_list[$((i % len_cuda))]}
    LAYER=${layer_list[i]}

    echo "initailizing worker evaluate MAESTRO at cuda ${cur_cuda_device}"
    log_name=data/MAESTRO/logs/eval.layer${LAYER}.log
    mkdir -p data/MAESTRO/logs

    CUDA_VISIBLE_DEVICES=${cur_cuda_device} \
    nohup \
    python . probe \
        --dataset MAESTRO \
        --pre_trained_model_type hubert \
        --pre_trained_folder data/hubert_data/HF_MuBERT_base_MPD-10Kh-MusicAll-900h-fma-full-noshit_HPO-v4_crop5s_ECD-v8_mix-v3-0.5_ckpt_136_250000 \
        --target_sr 24000 \
        --token_rate 75 \
        --audio_dir data/MAESTRO/hdf5s/maestro \
        --metadata_dir data/MAESTRO/maestro_target \
        --num_outputs 88 \
        --monitor valid_prec \
        --earlystop_patience 15 \
        --lr_scheduler_patience 5 \
        --max_epochs 200 \
        --batch_size 32 \
        --accumulate_grad_batches 2 \
        --token_rate 75 \
        --frame_threshold 0.5 \
        --lr 1e-3 \
        --layer ${LAYER} \
        --hidden_layer_sizes '[512]' \
        --num_workers 8 \
        --wandb_proj_name test \
        --wandb_run_name maestro_75hz_model_search_layer \
        --debug False \
        --auto_lr_find True \
    > ${log_name} 2>&1 &
    echo "worker evaluate MAESTRO at cuda ${cur_cuda_device} started"
    echo "see log at ${log_name}"
    sleep 1
done

exit