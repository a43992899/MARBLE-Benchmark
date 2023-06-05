import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
import wget

import benchmark.models.mule as mule
from benchmark.utils.audio_utils import load_audio, find_audios

def download_model():
    """Download the model from the repository."""
    links = mule.config.DOWNLOAD_URLS
    pretrain_folder = mule.config.PRETRAIN_FOLDER

    if not os.path.exists(pretrain_folder):
        print(f"Creating folder {pretrain_folder} for downloading pretrained mule checkpoint...""")
        os.makedirs(pretrain_folder)

    for name, link in links.items():
        download_path = os.path.join(pretrain_folder, name)
        if not os.path.exists(download_path):
            print(f"Downloading {name} from {link}...")
            download_dir = os.path.dirname(download_path)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir)
            wget.download(link, out=download_dir)

def select_args(config):
    args = argparse.Namespace()
    args.accelerator = config.dataset.pre_extract.accelerator
    args.output_dir = config.dataset.pre_extract.output_dir
    args.overwrite = config.dataset.pre_extract.overwrite
    args.audio_dir = config.dataset.pre_extract.audio_dir
    args.n_shard = config.args.n_shard
    args.shard_rank = config.args.shard_rank
    args.keep_folder_structure = config.dataset.pre_extract.keep_folder_structure
    return args
    
def main(config):
    args = select_args(config)

    download_model()

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')

    if args.n_shard > 1:
        print(f'processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort() # make sure no intersetction
        audio_files = audio_files[args.shard_rank * len(audio_files) // args.n_shard : (args.shard_rank + 1) * len(audio_files) // args.n_shard]
    
    mule_model = mule.mule.load_model(f'{mule.config.PRETRAIN_FOLDER}/supporting_data/model')

    for audio_file in tqdm(audio_files):
        # load audio
        try:
            waveform = load_audio(
                audio_file,
                target_sr=mule.config.SAMPLE_RATE,
                is_mono=True,
                is_normalize=False,
                crop_to_length_in_sec=None,
            )
        except Exception as e:
            print(f"skip audio {audio_file} because of {e}")
            continue
        
        # extract features
        waveform = waveform.squeeze().cpu().numpy()
        mel_data = mule.mule.to_mel(waveform)
        embeddings = mule.mule.extract_embeddings(mule_model, mel_data) # [1728, 15]
        embeddings = embeddings.mean(axis=1) # [1728]
        # reshape to [1, 1, 1728]
        out = embeddings.reshape(1, 1, -1)
        
        # save to npy
        if args.keep_folder_structure:
            output_file = os.path.join(
                args.output_dir,
                os.path.relpath(audio_file, args.audio_dir)+'.npy',
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        else:
            output_file = os.path.join(
                args.output_dir,
                os.path.basename(audio_file)+'.npy',
            )
        if not args.overwrite:
            assert not os.path.exists(output_file), f"{output_file} exists"
        np.save(output_file, out)
