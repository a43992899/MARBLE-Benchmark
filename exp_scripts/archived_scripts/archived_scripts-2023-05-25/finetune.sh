# TODO: finish header for this file, and update readme

python . finetune   --finetune False 
                    --dataset GTZAN --audio_dir data/GTZAN/genres --batch_size 4 --monitor valid_acc \
                    --hblayer None --hbreduction mean --probelayer all \
                    --pre_trained_folder ./data/hubert_data/HF_HuBERT_base_MPD_train_1000h_valid_300h_iter2_400k_vanilla_model_ncluster_2000_ckpt_54_200000
                    --sample_duration 30

