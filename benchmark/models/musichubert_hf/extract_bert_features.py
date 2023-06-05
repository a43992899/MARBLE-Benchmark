import os
import argparse

import numpy as np
import torch
from tqdm import tqdm

import benchmark as bench
from benchmark.utils.audio_utils import load_audio, find_audios


def select_args(config):
    args = argparse.Namespace()
    args.accelerator = config.dataset.pre_extract.accelerator
    args.output_dir = config.dataset.pre_extract.output_dir
    args.overwrite = config.dataset.pre_extract.overwrite
    args.audio_dir = config.dataset.pre_extract.audio_dir
    args.n_shard = config.args.n_shard
    args.shard_rank = config.args.shard_rank
    args.keep_folder_structure = config.dataset.pre_extract.keep_folder_structure
    args.pre_trained_folder = config.dataset.pre_extract.feature_extractor.pretrain.pre_trained_folder
    args.target_sr = config.dataset.pre_extract.feature_extractor.pretrain.target_sr
    args.force_half = config.dataset.pre_extract.feature_extractor.force_half
    args.processor_normalize = config.dataset.pre_extract.feature_extractor.pretrain.processor_normalize
    args.is_mono = config.dataset.pre_extract.audio_loader.is_mono
    args.is_normalize = config.dataset.pre_extract.audio_loader.is_normalize
    args.crop_to_length_in_sec = config.dataset.pre_extract.audio_loader.crop_to_length_in_sec
    args.crop_randomly = config.dataset.pre_extract.audio_loader.crop_randomly
    args.sliding_window_size_in_sec = config.dataset.pre_extract.audio_loader.sliding_window_size_in_sec
    args.sliding_window_overlap_in_percent = config.dataset.pre_extract.audio_loader.sliding_window_overlap_in_percent
    args.layer = config.dataset.pre_extract.feature_extractor.layer
    args.reduction = config.dataset.pre_extract.feature_extractor.reduction
    args.model_name = config.dataset.pre_extract.feature_extractor.pretrain.name
    args.huggingface_model_name = config.dataset.pre_extract.feature_extractor.pretrain.huggingface_model_name
    return args


def main(config):
    args = select_args(config)

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

    FeatureExtractor = eval(bench.NAME_TO_PRETRAIN_CLASS[args.model_name])
    feature_extractor = FeatureExtractor(
        args.pre_trained_folder if args.pre_trained_folder else args.huggingface_model_name,
        args.target_sr,
        args.force_half,
        processor_normalize=args.processor_normalize,
    )
    feature_extractor.to(device)
    feature_extractor.eval()
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            # load audio
            try:
                waveform = load_audio(
                    audio_file,
                    target_sr=args.target_sr,
                    is_mono=args.is_mono,
                    is_normalize=args.is_normalize,
                    crop_to_length_in_sec=args.crop_to_length_in_sec,
                    crop_randomly=args.crop_randomly,
                    device=device,
                )
            except Exception as e:
                print(f"skip audio {audio_file} because of {e}")
                continue
            
            # extract features
            # preprocess
            wav = feature_extractor.process_wav(waveform)
            wav = wav.to(device)
            # cut to 30s chunks
            if args.sliding_window_size_in_sec:
                assert args.sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
                overlap_in_sec = args.sliding_window_size_in_sec * args.sliding_window_overlap_in_percent / 100
                wavs = []
                for i in range(0, wav.shape[-1], int(args.target_sr * (args.sliding_window_size_in_sec - overlap_in_sec))):
                    wavs.append(wav[:, i : i + int(args.target_sr * args.sliding_window_size_in_sec)])
                # discard the last chunk if it is shorter than 1s
                if wavs[-1].shape[-1] < args.target_sr * 1:
                    wavs = wavs[:-1]
                features = []
                for wav in wavs:
                    features.append(feature_extractor(wav, layer=args.layer, reduction=args.reduction))
                features = torch.cat(features, dim=1) #.mean(dim=1, keepdim=True)
            else:
                features = feature_extractor(wav, layer=args.layer, reduction=args.reduction)

            # save to npy
            features = features.cpu().numpy()
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
