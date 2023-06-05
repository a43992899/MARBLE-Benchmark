"""
This script follows the jukemir protocol to extract musicnn features.
Which means using mean_pool + max_pool as the final feature.
"""
import os
import argparse

import torch
import numpy as np
from tqdm import tqdm
import wget

import benchmark.models.musicnn as musicnn
from benchmark.utils.audio_utils import load_audio, find_audios


def download_model():
    """Download the model from the repository."""
    links = musicnn.config.MSD_MUSICNN_BIG_LINKS
    pretrain_folder = musicnn.config.PRETRAIN_FOLDER

    if not os.path.exists(pretrain_folder):
        print(f"Creating folder {pretrain_folder} for downloading pretrained musicnn `MSD_MUSICNN_BIG` checkpoint...""")
        os.makedirs(pretrain_folder)

    for name, link in links.items():
        if not os.path.exists(os.path.join(pretrain_folder, name)):
            print(f"Downloading {name} from {link}...")
            wget.download(link, out=pretrain_folder)

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
    # disable eager execution for v1 compatibility
    musicnn.extractor.disable_eager_execution()

    if args.accelerator == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device(args.accelerator)

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')

    if args.n_shard > 1:
        print(f'processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort() # make sure no intersetction
        audio_files = audio_files[args.shard_rank * len(audio_files) // args.n_shard : (args.shard_rank + 1) * len(audio_files) // args.n_shard]

    # the original package does not support pre-load model, which is very slow
    PRE_LOAD_MODEL = True

    if PRE_LOAD_MODEL:
        sess, x, is_training, extract_vector, labels = \
                musicnn.extractor.load_model(musicnn.config.PRETRAIN_FOLDER)

    for audio_file in tqdm(audio_files):
        # load audio
        try:
            waveform = load_audio(
                audio_file,
                target_sr=musicnn.config.SR,
                is_mono=True,
                is_normalize=False,
                crop_to_length_in_sec=None,
            )
        except Exception as e:
            print(f"skip audio {audio_file} because of {e}")
            continue
        
        # extract features
        waveform = waveform.squeeze(0).numpy()
        if PRE_LOAD_MODEL:
            taggram, tags, features = musicnn.extractor.extractor_fast(
                waveform,
                sess, x, is_training, extract_vector, labels,)
        else:
            taggram, tags, features = musicnn.extractor.extractor(
                musicnn.config.PRETRAIN_FOLDER,
                waveform
            )
        
        out = np.concatenate(
            [features[k].mean(axis=0) for k in ["mean_pool", "max_pool"]]
        ) # [4194,] 
        # reshape to [1, 1, 4194]
        out = out.reshape(1, 1, -1)

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

    if PRE_LOAD_MODEL:
        sess.close()
