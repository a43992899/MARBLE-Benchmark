#!/usr/bin/bash

# example: 
# bash exp_scripts/convert_HuBERT_HF_script.sh  baai_sync_to_shef music_hubert HPO-v2-based_LogMel-200-Chroma-300_400k 216_400000
# bash exp_scripts/convert_HuBERT_HF_script.sh  baai_sync_to_shef music_hubert HPO-v2-based_LogMel-200-Chroma-300_400k 216_400000
# example with additional 
# short name: bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-200-Chroma-300_400k 216_400000 HPO-v2-based_L200C300_400k
# short name: bash exp_scripts/convert_HuBERT_HF_script.sh baai_sync_to_shef music_hubert HPO-v2-based_LogMel-100-Chroma-400_400k 216_400000 HPO-v2-based_L100C400_400k
{
    CONVERT_SETTING=${1:-'baai_sync_to_shef'} # baai_sync_to_shef, local_test_shef, local_test_baai
    HUBERT_TYPE=${2:-'music_hubert'} # hubert music_hubert

    # ---------- setting for MPD 1000H ----------
    # training_setting=HuBERT_base_MPD_train_1000h_valid_300h_iter2_400k

    
    model_name=${3:-'vanilla_Chroma-264-1_hier-ensemble-v1'}

    # model_name=vanilla_model_Ensemble-2_MFCC-13-3_K-300_CQT-84-3_K-200
    # model_name=vanilla_model_Ensemble-2_LogMel-229-1_K-300_Chroma-264-1_K-200_50Hz
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.1_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.5_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-sample-0.1_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.1_ORI-0
    # model_name=vanilla_model_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_alpha-0.75
    # model_name=vanilla_Chroma-264-1_hier-ensemble-v1

    # model_name=Ensemble-2_LogMel-229-1-300_Chroma-264-1-200_L6_K-500
    # model_name=Ensemble-2_LogMel-229-1-300_Chroma-264-1-200_L6_K-250_L11_K-250
    # model_name=Ensemble-2_LogMel-229-1-300_Chroma-264-1-200_L11_K-500

    # model_name=vanilla_model_Ensemble-2_LogMel-229-1_K-300_Chroma-264-1_K-200_50Hz_opt1
    # model_name=vanilla_model_Ensemble-2_LogMel-229-1_K-300_Chroma-264-1_K-200_50Hz_opt2 

    # model_name=vanilla_model_Chroma-264-1_K-500_50Hz
    # model_name=vanilla_model_LogMel-229-1_K-500_50Hz
    # model_name=d2v_arc_LogMel-K-300-Chroma-K-200

    # model_name=base_LogMel-K-300-Chroma-K-200_T-cqt-m-1
    # model_name=base_LogMel-K-300-Chroma-K-200_T-cqt-m-4
    # model_name=base_LogMel-K-300-Chroma-K-200_T-cqt-m-1-bin-336-L1

    # model_name=base_LogMel-K-300-Chroma-K-200_mask0.65
    # model_name=base_LogMel-K-300-Chroma-K-200_mask0.5
    # model_name=LogMel-K-300-Chroma-K-200_m-0.8-5
    # model_name=LogMel-K-300-Chroma-K-200_m0.9-10

    # model_name=vanilla_LogMel-K-300-Chroma-K-200_Crop-15s
    # model_name=vanila_LogMel-K-300-Chroma-K-200_mask-dynamic-v1
    # model_name=vanila_LogMel-K-300-Chroma-K-200_mask-len-dynamic-v1
    # model_name=vanila_LogMel-K-300-Chroma-K-200_mask-len-dynamic-v2

    # model_name=vanila_LogMel-K-300-Chroma-K-200_cqt-feat-336_v1
    # model_name=vanila_LogMel-K-300-Chroma-K-200_cqt-feat-336_v2
    # model_name=vanila_LogMel-K-300-Chroma-K-200_cqt-feat-336_v3

    # model_name=vanilla_jukebox_5b_K-2048_v1

    # model_name=HPO_LogMel-300-Chroma-200_crop-15s_m-prob-0.5-len-5_cqt-pred-1.0_baseline-v2

    ckpt_step=${4:-'216_400000'}

    # ckpt_step=7_25000
    # ckpt_step=14_50000
    # ckpt_step=21_75000
    # ckpt_step=27_100000
    # ckpt_step=34_125000
    # ckpt_step=41_150000
    # ckpt_step=47_175000
    # ckpt_step=54_200000
    # ckpt_step=61_225000
    # ckpt_step=67_250000
    # ckpt_step=86_320000
    # ckpt_step=107_400000

    # for crop-15s
    # ckpt_step=14_25000
    # ckpt_step=27_50000
    # ckpt_step=54_100000
    # ckpt_step=134_250000

    # if not specify, then use original model name
    short_model_name=${5:-${model_name}} 

    training_setting=${6:-'HuBERT_base_MPD_train_1000h_iter1'}
    config_name=${7:-'config_musichubert'}

    case $CONVERT_SETTING in
        local_test_shef)
            # training_setting=HuBERT_base_MPD_train_1000h_valid_300h_iter1

            FAIRSEQ_PATH=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/src/fairseq
            project_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/
            
            fairseq_ckpt_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_1000h_HuBERT_pretrain
            fairseq_ckpt_folder=${fairseq_ckpt_parent_folder}/ckpt_${training_setting}/${model_name}
            checkopint_file_path=${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt
            
            output_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints/WaveEncoder

            shortcut_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts
            ;;
        local_test_baai)
            # training_setting=HuBERT_base_MPD_train_10Kh_iter1
            # training_setting=HuBERT_base_MPD_train_1000h_iter1

            FAIRSEQ_PATH=/share/project/music/music/MAP_benchmark/src/fairseq
            project_folder=/share/project/music/music/MAP_benchmark
            
            fairseq_ckpt_parent_folder=/share/project/music/music/fairseq_savedir/WaveEncoder_MPD_10Kh_HuBERT_pretrain
            fairseq_ckpt_folder=${fairseq_ckpt_parent_folder}/ckpt_${training_setting}/${model_name}
            checkopint_file_path=${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt
            
            output_parent_folder=/share/project/music/music/hf_ckpt/WaveEncoder

            shortcut_dir=/share/project/music/music/hf_ckpt_short
            ;;
        baai_sync_to_shef)
            # training_setting=HuBERT_base_MPD_train_1000h_iter1

            FAIRSEQ_PATH=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/src/fairseq
            project_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/
            
            fairseq_ckpt_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_ckpt_baai/checkpoint_sync
            # fairseq_ckpt_folder=${fairseq_ckpt_parent_folder}
            checkopint_file_path=${fairseq_ckpt_parent_folder}/${model_name}_checkpoint_${ckpt_step}.pt

            output_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints/WaveEncoder

            shortcut_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts

            ;;
        *)
            echo "Unknown setting: $CONVERT_SETTING"
            exit 1
            ;;
    esac

    case $HUBERT_TYPE in
        hubert)
            # # original HuBERT
            config_path=${project_folder}/src/convert_HF/HuBERT_base/config.json
            # custom_model_source_code=${FAIRSEQ_PATH}/examples/data2vec/models/data2vec_audio.py
            # custom_model_soft_link=${FAIRSEQ_PATH}/fairseq/models/data2vec_audio.py
            ;;
        music_hubert)
            # data2vec architecture
            # config_path=${project_folder}/src/convert_HF/HuBERT_base/config_d2v.json
            # customized model
            # config_path=${project_folder}/src/convert_HF/HuBERT_base/config_musichubert.json
            # config_path=${project_folder}/src/convert_HF/HuBERT_base/config_musichubert_cqt_feat.json
            config_path=${project_folder}/src/convert_HF/HuBERT_base/${config_name}.json
            custom_model_source_code=${FAIRSEQ_PATH}/examples/music_hubert/models/music_hubert
            custom_model_soft_link=${FAIRSEQ_PATH}/fairseq/models/music_hubert
            custom_dataset_source_code=${FAIRSEQ_PATH}/examples/music_hubert/data/music_hubert_dataset.py
            custom_dataset_soft_link=${FAIRSEQ_PATH}/fairseq/data/music_hubert_dataset.py
            custom_aug_source_code=${FAIRSEQ_PATH}/examples/music_hubert/data/music_data_augmentation
            custom_aug_soft_link=${FAIRSEQ_PATH}/fairseq/data/music_data_augmentation
            custom_task_source_code=${FAIRSEQ_PATH}/examples/music_hubert/tasks/music_hubert_pretraining.py
            custom_task_soft_link=${FAIRSEQ_PATH}/fairseq/tasks/music_hubert_pretraining.py
            ;;
        *)
            echo "Unknown HuBERT
             model type: $HUBERT_TYPE"
            exit 1
            ;;
    esac
    
    echo "remove temporary file ${custom_model_soft_link}"
    rm ${custom_model_soft_link}
    rm  ${custom_dataset_soft_link} ${custom_aug_soft_link}
    rm ${custom_task_soft_link}

    echo "temporary link to user defined models ${custom_model_soft_link}"
    ln -s ${custom_model_source_code} ${custom_model_soft_link}
    ln -s ${custom_dataset_source_code} ${custom_dataset_soft_link}
    ln -s ${custom_aug_source_code} ${custom_aug_soft_link}
    ln -s ${custom_task_source_code} ${custom_task_soft_link}


    cd ${project_folder}

    long_name=HF_${training_setting}_${model_name}_ckpt_${ckpt_step}
    output_folder=${output_parent_folder}/${long_name}
    
    short_name=HF_${short_model_name}_ckpt_${ckpt_step::-3}k
    shortcut_path=${shortcut_dir}/${short_name}
    
    echo loading from:
    echo  ${checkopint_file_path}
    echo output to:
    echo ${output_folder}
# exit
    mkdir -p ${output_folder}

    # for ckpt_step in 7_25000 14_50000 21_75000 27_100000 34_125000 41_150000 47_175000 54_200000 61_225000 67_250000
    # for ckpt_step in 7_25000 14_50000 21_75000 34_125000 41_150000 47_175000 54_200000 61_225000
    # do
    {
        # python HuBERT_convert.py --pytorch_dump_folder ${output_folder} --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt --config_path ./HuBERT_base/config.json --not_finetuned
        # python HuBERT_convert.py --pytorch_dump_folder ${output_folder} --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt --config_path ./HuBERT_base/config_d2v.json --not_finetuned


        python -m convert_HF.MusicHuBERT_convert --pytorch_dump_folder ${output_folder} --checkpoint_path ${checkopint_file_path} --config_path ${config_path} --not_finetuned

        echo "remove temporary file ${custom_model_soft_link}"
        rm ${custom_model_soft_link}
        rm  ${custom_dataset_soft_link} ${custom_aug_soft_link}
        rm ${custom_task_soft_link}

        # python HuBERT_convert.py --pytorch_dump_folder ${output_folder} \
        # --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt \
        # --config_path ./HuBERT_base/config.json --not_finetuned \
        # --user_dir ${FAIRSEQ_PATH}/examples/data2vec 

        # gdrive list -q 'name="MusicAudioPretrain_huggingface_checkpoints"'
        case $CONVERT_SETTING in
            local_test_baai)
                echo "please mannually sync the checkpoint saved at: ${output_parent_folder}"
                ;;
            *)
                echo "upload checkpoints"
                gdrive upload -p 1aDlnMWskvqyF-WSzk9jBfHsabP8O1bGy --recursive ${output_folder}
                ;;        
        esac

        echo "add short name link to ${shortcut_path}" 
        ln -s ${output_folder} ${shortcut_path}
        echo ${long_name},${short_name} >> ${project_folder}/exp_scripts/HF_ckpt_name_mapping.csv

    }
    # done
    exit
}