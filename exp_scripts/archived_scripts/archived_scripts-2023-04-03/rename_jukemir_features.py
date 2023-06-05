import glob
import os
import json
import shutil


data_root='/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR_Benchmark_Data/jukemir_representations'

datasets=[ 'giantsteps_clips', 'emomusic'] # 'magnatagatune', 'gtzan_ff',

features=['choi', 'chroma','clmr', 'jukebox','mfcc', 'musicnn']

with open('/mnt/fastdata/acp21aao/MusicAudioPretrain_project_dir/jukemir/cache_dir/processed/magnatagatune/meta.json') as f:
    mtt_mapping = json.load(f)
with open('/home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/jukemir/cache_dir/processed/gtzan_ff/meta.json') as f:
    gtzan_mapping = json.load(f)

for dataset in datasets:
    for feature in features:
        # feature_folder = os.path.join(data_root, dataset, f'{feature}')
        feature_folder = os.path.join(data_root, dataset, f'{feature}_feature_layer_all_reduce_mean')
        print(feature_folder)
        files = glob.glob(os.path.join(feature_folder,'**/*.npy'), recursive=True)
        # print(files[:3])
        # exit()

        for fn in files:
            # if '.wav' not in fn:
            if dataset == 'magnatagatune':
                uid=os.path.splitext(os.path.basename(fn))[0].replace('.wav','')
                target_path = mtt_mapping[uid]["extra"]["mp3_path"]
                target_subfolder = os.path.join(feature_folder, target_path.split('/')[0])
                os.makedirs(target_subfolder, exist_ok=True)

                shutil.move(fn, os.path.join(feature_folder,target_path)+'.npy')
                # new_name = fn.replace('.npy','.mp3.npy')

            elif dataset == 'gtzan_ff':
                uid=os.path.splitext(os.path.basename(fn))[0].replace('.wav','')
                target_subfolder = os.path.join(feature_folder, gtzan_mapping[uid]["y"])

                # print(target_subfolder, fn)
                # exit()
                os.makedirs(target_subfolder, exist_ok=True)

                target_name = gtzan_mapping[uid]["extra"]["id"]
                shutil.move(fn, os.path.join(target_subfolder, target_name + '.wav.npy'))
                
            else:
                # new_name = fn.replace('.npy','.wav.npy')
                

                new_name = fn.replace('.wav.wav.npy','.wav.npy')
                # print(new_name)
                # print(fn)
                # exit()
                os.rename(fn, new_name)

# for fn in `ls *.tar.gz`; do tar -xzvf ${fn}; done
# ln -s /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR_Benchmark_Data/jukemir_representations/magnatagatune data/MTT/jukemir_features
# ln -s /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR_Benchmark_Data/jukemir_representations/gtzan_ff data/GTZAN/jukemir_features
# ln -s /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR_Benchmark_Data/jukemir_representations/giantsteps_clips data/GS/jukemir_features
# ln -s /home/acp21aao/my_fastdata/MusicAudioPretrain_project_dir/MIR_Benchmark_Data/jukemir_representations/emomusic data/EMO/jukemir_features

# for fd in `ls -d data/*/jukemir_features/*`; do echo ${fd}; mv ${fd} ${fd}_feature_layer_all_reduce_mean; done 