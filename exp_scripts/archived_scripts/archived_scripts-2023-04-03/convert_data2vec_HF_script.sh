#!/bin/bash

# example: 
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_crop10s 1240_400000
# bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef data2vec_audio d2v_vanila_prob50 378_400000
# example with additional 
# short name: bash exp_scripts/convert_data2vec_HF_script.sh  baai_sync_to_shef d2v_vanila_crop10s 1240_400000 d2v_crop10s

{
    CONVERT_SETTING=${1:-'baai_sync_to_shef'} # baai_sync_to_shef, local_test_shef
    D2V_TYPE=${2:-'data2vec_audio'} # data2vec_audio data2vec_music

    # ---------- setting for MPD 1000H ----------
    training_setting=data2vec_base_MPD_train_1000h
    # model_name=vanilla_model_Ensemble-2_MFCC-13-3_K-300_CQT-84-3_K-200
    # model_name=vanilla_model_Ensemble-2_LogMel-229-1_K-300_Chroma-264-1_K-200_50Hz 
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.1_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.5_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-sample-0.1_ORI-0.1
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.1_ORI-0
    # model_name=vanilla_model_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_alpha-0.75

    # model_name=vanila_base
    # model_name=lr_opt_v2
    # model_name=tune_HuBERT_v3
    # model_name=d2v_HuBERT_arc_base
    # model_name=d2v_ditsill_HuBERT_v1
    # model_name=d2v_vanila_cqt-pred-v9

    model_name=${3:-'d2v_vanila_cqt-pred-v9'}

    # ckpt_step=46_22745
    # ckpt_step=41_40000
    # ckpt_step=81_80000
    # ckpt_step=122_120000
    # ckpt_step=189_200000
    # ckpt_step=283_280000
    # ckpt_step=405_400000
    ckpt_step=${4:-'405_400000'}

    # if not specify, then use original model name
    short_model_name=${5:-${model_name}} 

    echo "using short model name for HF checkpoint: ${short_model_name}"
    # exit

    case $CONVERT_SETTING in
        local_test_shef)
            FAIRSEQ_PATH=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/src/fairseq
            project_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MusicAudioPretrain/
            
            fairseq_ckpt_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_1000h_data2vec_pretrain
            fairseq_ckpt_folder=${fairseq_ckpt_parent_folder}/ckpt_${training_setting}/${model_name}
            checkopint_file_path=${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt
            
            output_parent_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints/WaveEncoder

            shortcut_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts
            ;;
        baai_sync_to_shef)
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

    case $D2V_TYPE in
        data2vec_audio)
            custom_model_source_code=${FAIRSEQ_PATH}/examples/data2vec/models/data2vec_audio.py
            custom_model_soft_link=${FAIRSEQ_PATH}/fairseq/models/data2vec_audio.py
            ;;
        data2vec_music)
            custom_model_source_code=${FAIRSEQ_PATH}/examples/data2vec/models/data2vec_music.py
            custom_model_soft_link=${FAIRSEQ_PATH}/fairseq/models/data2vec_music.py
            ;;
        *)
            echo "Unknown data2vec model type: $D2V_TYPE"
            exit 1
            ;;
    esac
    echo "temporary link to user defined models ${custom_model_soft_link}"
    ln -s ${custom_model_source_code} ${custom_model_soft_link}

    cd ${project_folder}

    long_name=HF_${training_setting}_${model_name}_ckpt_${ckpt_step}
    output_folder=${output_parent_folder}/${long_name}
    
    short_name=HF_d2v_base_MPD_train_1000h_${short_model_name}_ckpt_${ckpt_step::-3}k
    shortcut_path=${shortcut_dir}/${short_name}

    # echo "listing checkpoints in  ${fairseq_ckpt_folder}"
    # ls ${fairseq_ckpt_folder}

    # # ---------- setting for MPD 10000H ----------
    # training_setting=HuBERT_base_MPD_train_10000h_valid_300h_iter1_250k
    # model_name=vanilla_Ensemble-2_MFCC-13-3_K-300_Chroma-24-1_K-200_RPL-in-batch-0.1_ORI-0.1
    # # ckpt_step=3_100000
    # ckpt_step=7_250000
    # output_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_10000h_HuBERT_pretrain/HF_checkpoints/HF_${training_setting}_${model_name}_ckpt_${ckpt_step}
    # fairseq_ckpt_folder=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_10000h_HuBERT_pretrain/None/ckpt_${training_setting}/${model_name}

    echo loading from:
    echo  ${checkopint_file_path}
    echo output to:
    echo ${output_folder}

    mkdir -p ${output_folder}

    # python -c "from transformers import Data2VecAudioConfig; config = Data2VecAudioConfig.from_pretrained('facebook/data2vec-audio-base-960h'); config.save_pretrained('./data2vec_base');"
    # python -c "from transformers import Data2VecAudioConfig; config = Data2VecAudioConfig.from_pretrained('facebook/data2vec-audio-base-960h'); config.save_pretrained('./data2vec_base');"

    # eval "python data2vec_convert.py --pytorch_dump_folder ${output_folder} --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt --config_path ./data2vec_base/config.json --not_finetuned"




    # huggingface config 

    # # original HuBERT
    config_path=${project_folder}/src/convert_HF/data2vec_base/config.json
    # config_path=${project_folder}/src/convert_HF/data2vec_base/config_tune_HuBERT.json

    # python data2vec_convert.py --pytorch_dump_folder ${output_folder} --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt \
    # --config_path ${config_path} \
    # --model_type data2vec \
    # --not_finetuned

    python -m convert_HF.data2vec_convert --pytorch_dump_folder ${output_folder} --checkpoint_path ${checkopint_file_path} \
    --config_path ${config_path} \
    --model_type data2vec \
    --not_finetuned

    # python data2vec_convert.py --pytorch_dump_folder ${output_folder} --checkpoint_path ${fairseq_ckpt_folder}/checkpoint_${ckpt_step}.pt \
    # --config_path ./data2vec_base/config_tune_HuBERT.json \
    # --model_type HuBERT \
    # --not_finetuned

    echo "remove temporary file ${custom_model_soft_link}"
    rm ${custom_model_soft_link}

    # gdrive list -q 'name="MusicAudioPretrain_huggingface_checkpoints"'
    # upload checkpoints
    gdrive upload -p 1aDlnMWskvqyF-WSzk9jBfHsabP8O1bGy --recursive ${output_folder}

    # nohup gdrive upload -p 1Maf61iJX4jyZt5bOyTxis57OvubFWoBL --recursive  /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_1000h_HuBERT_pretrain/None/ckpt_HuBERT_base_MPD_train_1000h_valid_300h_iter1_250k > gdrive.upload.hubert_base.log 2>&1 & 
    # nohup gdrive upload -p 1MqBYe1oOaM2R2JzVCFDCxFAAkqfweyh5 --recursive  /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/fairseq_savedir/WaveEncoder_MPD_1000h_data2vec_pretrain/ckpt_data2vec_base_MPD_train_1000h_400k > gdrive.upload.data2vec_base.log 2>&1 & 
    # 

    echo "add short name link to ${shortcut_path}" 
    ln -s ${output_folder} ${shortcut_path}

    echo ${long_name},${short_name} >> ${project_folder}/exp_scripts/HF_ckpt_name_mapping.csv

    exit
}