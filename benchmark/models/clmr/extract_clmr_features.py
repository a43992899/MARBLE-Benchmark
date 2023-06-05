"""
This script follows the jukemir protocol to extract clmr features.
"""
import os
import argparse

import wget
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from benchmark.utils.audio_utils import load_audio, find_audios
import benchmark.models.clmr as clmr


def download_model():
    if not os.path.exists(clmr.config.PRETRAIN_FOLDER):
        os.makedirs(clmr.config.PRETRAIN_FOLDER)
        print(f"Created folder: {clmr.config.PRETRAIN_FOLDER}")

    if not os.path.exists(f"{clmr.config.PRETRAIN_FOLDER}/clmr_magnatagatune_mlp/clmr_epoch=10000.ckpt"):
        wget.download(clmr.config.ZIP_URL, clmr.config.PRETRAIN_FOLDER)
        os.system(f"unzip -o {clmr.config.PRETRAIN_FOLDER}/clmr_magnatagatune_mlp.zip -d {clmr.config.PRETRAIN_FOLDER}")
        print(f"Downloaded model to {clmr.config.PRETRAIN_FOLDER}")


class CLMRFeature(nn.Module):
    def __init__(
            self,
            pre_trained_folder,
        ) -> None:
        super().__init__()
        self.encoder = clmr.clmr.SampleCNN(
            strides=[3, 3, 3, 3, 3, 3, 3, 3, 3],
            supervised=0,
            out_dim=50,
        )

        state_dict = clmr.clmr.load_encoder_checkpoint(f"{pre_trained_folder}/clmr_magnatagatune_mlp/clmr_epoch=10000.ckpt", 50)
        self.encoder.load_state_dict(state_dict)
        self.encoder.fc = clmr.clmr.Identity()
        self.sr = clmr.config.SAMPLE_RATE

    def forward(self, input_values):
        out = self.encoder(input_values)
        return out.mean(dim=0)


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

    feature_extractor = CLMRFeature(
        clmr.config.PRETRAIN_FOLDER,
    )
    feature_extractor.eval()
    feature_extractor.to(device)

    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            # load audio
            try:
                waveform = load_audio(
                    audio_file,
                    target_sr=clmr.config.SAMPLE_RATE,
                    is_mono=True,
                    is_normalize=False,
                    crop_to_length_in_sec=None,
                    device=device,
                )
            except Exception as e:
                print(f"skip audio {audio_file} because of {e}")
                continue
            
            # extract features
            wavs = []
            wav = waveform
            for i in range(0, wav.shape[-1], int(clmr.config.SAMPLE_RATE * 30)):
                wavs.append(wav[:, i : i + int(clmr.config.SAMPLE_RATE * 30)])
            if wavs[-1].shape[-1] < clmr.config.SAMPLE_RATE * 1:
                wavs = wavs[:-1]
            
            all_features = []
            for wav in wavs:
                frames = torch.split(wav, clmr.config.FRAME_LENGTH, dim=1)
                if len(frames) <= 1:
                    continue
                frames = torch.cat(frames[:-1], dim=0)
                frames = frames.unsqueeze(dim=1)
                with torch.no_grad():
                    features = feature_extractor(frames.to(device)) # [512]
                # reshape to [1, 1, 512]
                features = features.reshape(1, 1, -1)
                features = features.cpu().numpy()
                all_features.append(features)
            
            features = np.concatenate(all_features, axis=1)
            features = features.mean(axis=1, keepdims=True) # [1, 1, 512]
            
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
                assert not os.path.exists(output_file), f"{output_file} exists. If you want to overwrite, please add --overwrite."
            np.save(output_file, features)
