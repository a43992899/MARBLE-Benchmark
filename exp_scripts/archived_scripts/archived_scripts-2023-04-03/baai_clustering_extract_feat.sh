
{   
    # example: bash /sharefs/music/music/MusicAudioPretrain/exp_scripts/baai_clustering_extract_feat.sh baai MFCC 13-3 64 8
    # example: bash /sharefs/music/music/MusicAudioPretrain/exp_scripts/baai_clustering_extract_feat.sh baai Chroma 264-1 64 8
    # example: bash /sharefs/music/music/MusicAudioPretrain/exp_scripts/baai_clustering_extract_feat.sh baai LogMel 229-1 32 8
    PLATFORM=${1:-'baai'}
    feat_type=${2:-'MFCC'}
    dimension=${3:-'13-3'}
    train_nshard=${4:-'32'}
    valid_nshard=${5:-'8'}

    source /home/zhangge/.bashrc
    source /opt/conda/etc/profile.d/conda.sh
    conda activate /home/zhangge/.conda/envs/map
    FAIRSEQ_PATH=/sharefs/music/music/MusicAudioPretrain/src/fairseq
    
    export all_proxy=http://httpproxy-headless.kubebrain:3128 no_proxy=platform.wudaoai.cn,platform.baai.ac.cn,kubebrain,kubebrain.com,svc,brainpp.cn,brainpp.ml,127.0.0.1,localhost; export http_proxy=$all_proxy https_proxy=$all_proxy

    cd /sharefs/music/music/MusicAudioPretrain/src/clustering
    bash run_get_audio_feat.sh ${PLATFORM} ${feat_type} ${dimension} ${train_nshard} ${valid_nshard}

exit
}