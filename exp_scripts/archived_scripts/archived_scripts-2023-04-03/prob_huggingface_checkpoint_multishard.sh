#!/bin/bash
# example: bash exp_scripts/prob_huggingface_checkpoint.sh hubert
{
    MODEL_TYPE=${1:-'hubert'} # hubert, data2vec
    ENV_SETTING=${2:-'local_test_baai'}
    MODEL_NAME=$3
    CKPT_STEP=$4
    MODEL_SETTING=${5:-'12_768'}
    TARGET_SAMPLE_RATE=${6:-'16000'}
    TEST_DATASET=${7:-"GTZAN,GS,MTT,EMO"}
    ENABLE_EXTRACT=${8:-'true'}
    ENABLE_PROBE=${9:-'true'}
    N_SHARD=${10:-'4'}
    DEVICE_LIST=${11-'0,1,2,3,4,5,6,7'} # only for multiple-accelerator feature extraction
    PROCESSOR_NORMALIZE=${12:-'True'}
    
    case $ENV_SETTING in
        local_test_default)
            project_dir=./
            ckpt_dir=./data/hubert_data
            ACCELERATOR=gpu
            ACC_PRECISION=16
            ;;
        local_test_shef)
            project_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MAP_benchmark
            ckpt_dir=/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/huggingface_checkpoints_shorts
            ACCELERATOR=gpu
            ACC_PRECISION=16
            ;;
        local_test_baai)
            # project_dir=/share/project/music/music/MAP_benchmark
            # ckpt_dir=/share/project/music/music/hf_ckpt_short
            project_dir=./
            ckpt_dir=./data/hubert_data
            ACCELERATOR=mlu
            ACC_PRECISION=32
            # export WANDB_API_KEY=bf72b4f3341b5e72073c1ad84856802f67703ed6
            export WANDB_DIR=/share/project/music/music/wandb
            # export WANDB_MODE=offline
            export WANDB_MODE=online
            export WANDB_RESUME=allow
            export HYDRA_FULL_ERROR=1
            # source /home/zhangge/.bashrc
            # source /opt/conda/etc/profile.d/conda.sh
            # conda activate /home/zhangge/.conda/envs/map_eval
            # conda list
            # ulimit -Sn 900000
            # ulimit -Sn unlimited
            # ulimit -a
            # export all_proxy=http://httpproxy-headless.kubebrain:3128 no_proxy=platform.wudaoai.cn,platform.baai.ac.cn,kubebrain,kubebrain.com,svc,brainpp.cn,brainpp.ml,127.0.0.1,localhost; export http_proxy=$all_proxy https_proxy=$all_proxy
            # FAIRSEQ_PATH=/share/project/music/music/MAP_benchmark/src/fairseq;

            ;;
        *)
            echo "Unknown setting: $ENV_SETTING"
            exit 1
            ;;
    esac

cd ${project_dir}
echo ${project_dir}
# exit 
OUTPUT_FEAT_ROOT=./data


for HF_CHECKPOINT_DIR in `ls -d ${ckpt_dir}/*_${MODEL_NAME}_ckpt_*_${CKPT_STEP}`;
do 
    TEST_DATASET=${TEST_DATASET//,/ }
    for dataset in ${TEST_DATASET}; do {

        if [ ${ENABLE_EXTRACT} == 'true' ]; then
            echo "extracting checkpoint feature at ${HF_CHECKPOINT_DIR} for dataset ${dataset}, model type ${MODEL_TYPE}, audio sample rate ${TARGET_SAMPLE_RATE}"
            bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} \
                ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${dataset} ${TARGET_SAMPLE_RATE} \
                ${N_SHARD} ${DEVICE_LIST} ${PROCESSOR_NORMALIZE} \
                ${ACCELERATOR}
        fi
        
        if [ ${ENABLE_PROBE} == 'true' ]; then
            echo "probing checkpoint at ${HF_CHECKPOINT_DIR} for dataset ${dataset}, model type ${MODEL_TYPE}, audio sample rate ${TARGET_SAMPLE_RATE}"
            bash exp_scripts/probe_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} \
                ${OUTPUT_FEAT_ROOT} ${dataset} ${MODEL_SETTING} \
                ${ACCELERATOR} ${ACC_PRECISION}
        fi
    }
    done
done
# done
exit
}
