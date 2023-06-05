import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm
from nnAudio import features as nnAudioFeatures
import librosa

from benchmark.utils.audio_utils import load_audio, find_audios


from multiprocessing import Pool
from functools import partial

class HandCraftedFeature(nn.Module):

    def __init__(
            self,
            sample_rate,mode
        ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mode=mode
        self.mfcc_ceps=13
        self.n_chroma=264
        self.n_fft = 512
        self.label_rate=50
        self.win_length = sample_rate//400 # 400 if 16k
        self.hop_length = sample_rate//self.label_rate # label_rate = 50Hz or 100Hz
        self.spec_layer = nnAudioFeatures.cqt.CQT(sr=sample_rate, hop_length=self.hop_length, fmin=32.7, 
                        fmax=None, n_bins=336, bins_per_octave=336//7,
                        filter_scale=1, norm=1, window='hann',center=True, 
                        pad_mode='constant', trainable=False, 
                        output_format='Magnitude', verbose=True)

        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

    @torch.no_grad()
    def process_wav(self, waveform):
        # return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",    
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]

    @torch.no_grad()
    def __call__(self, input_values):  # weighted 
        if self.mode=='mfcc':
            x = input_values.view(1, -1)
            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
                num_ceps=self.mfcc_ceps,
                frame_shift=1000/50, 
            )  # (time, freq)
            mfccs = mfccs.transpose(0, 1)  # (freq, time)
            deltas = torchaudio.functional.compute_deltas(mfccs)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
            concat = concat.transpose(0, 1).contiguous() # (n token, dim)
            return concat.cpu().numpy()

        elif self.mode=='cqt':
            # Initializing the model
            spec = self.spec_layer(input_values)[0] # Feed-forward your waveform to get the spectrogram
            return spec.transpose(0, 1).cpu().numpy()
        elif self.mode=='chroma':
            x=input_values.cpu().numpy()
            m = 1 + (x.shape[1] - self.win_length) // self.hop_length
            chroma = librosa.feature.chroma_stft(
                x[:m*self.hop_length + (self.win_length - self.hop_length)], 
                sr=self.sample_rate, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length, 
                win_length=self.win_length,
                n_chroma=self.n_chroma
            )
            chroma = chroma[0, :, :-2]
            # print(chroma.shape)
            return np.transpose(chroma,(1,0)) # (n token, dim)

class HandCraftedFeature_Jukemir(nn.Module):

    def __init__(
            self,
            sample_rate,mode
        ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.mode=mode
        self.label_rate=50
        self.win_length = sample_rate//400 # 400 if 16k
        self.hop_length = sample_rate//self.label_rate # label_rate = 50Hz or 100Hz
        self.spec_layer = nnAudioFeatures.cqt.CQT(sr=sample_rate, hop_length=self.hop_length, fmin=32.7, 
                        fmax=None, n_bins=336, bins_per_octave=336//7,
                        filter_scale=1, norm=1, window='hann',center=True, 
                        pad_mode='constant', trainable=False, 
                        output_format='Magnitude', verbose=True)

        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

    @torch.no_grad()
    def process_wav(self, waveform):
        # return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",    
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]

    @torch.no_grad()
    def __call__(self, input_values): 
        if self.mode=='mfcc':
            x = input_values.numpy()
            features = librosa.feature.mfcc(y=x, sr=self.sample_rate, hop_length=self.hop_length, n_mfcc=20)
            features = features[0, :, :-2]
            features = np.transpose(features,(1,0))

        elif self.mode=='cqt':
            features = self.spec_layer(input_values)[0]
            features = features.transpose(0, 1).cpu().numpy()

        elif self.mode=='chroma':
            x = input_values.numpy()
            m = 1 + (x.shape[1] - self.win_length) // self.hop_length
            features = librosa.feature.chroma_cqt(
                y=x[:m*self.hop_length + (self.win_length - self.hop_length)], 
                sr=self.sample_rate, 
                hop_length=self.hop_length,
                n_chroma=12
            )
            features = features[0, :, :-2]
            features = np.transpose(features,(1,0)) # (n token, dim)
        else:
            raise NotImplementedError
        
        # Stack differences/statistics
        moments = []
        for i in range(3):
            f = np.diff(features, n=i, axis=0)
            moments.append(f.mean(axis=0))
            moments.append(f.std(axis=0))
        moments = np.concatenate(moments)

        return moments


def extract_audio_feat(audio_file, feature_extractor, device, args):
    # load audio
    with torch.no_grad():
        try:
            waveform = load_audio(
                audio_file,
                target_sr=args.target_sr,
                is_mono=args.is_mono,
                is_normalize=args.is_normalize,
                crop_to_length_in_sec=args.crop_to_length_in_sec
            )
        except Exception as e:
            print(f"skip audio {audio_file} because of {e}")
            # continue
            return 1
        # extract features
        input_values = feature_extractor.process_wav(waveform)
        input_values = input_values.to(device)
        features = feature_extractor(input_values)

        # save to npy
        features = features.reshape((1, 1, -1))
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
        np.save(output_file, features)
        return 0

def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    print(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end

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
    
    args.feature_type = args.feature_type.lower()
    if torch.cuda.is_available() and args.feature_type == 'cqt': 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')
    
    audio_files.sort()# make sure the order of shardings
    if args.nshard > 0 and args.shard_rank >= 0:
        start, end = get_shard_range(len(audio_files), args.nshard, args.shard_rank)
        audio_files = audio_files[start:end]

    feature_extractor = HandCraftedFeature_Jukemir(
        args.target_sr,mode=args.feature_type
    )
    feature_extractor.to(device)
    feature_extractor.eval()

    # if args.n_proc <= 1:
    for audio_file in tqdm(audio_files):
        extract_audio_feat(audio_file, feature_extractor, device, args)
    # else:
    #     # tqdm version multiprocessing
    #     print(f'Extracting with {args.n_proc} processes')
    #     with Pool(args.n_proc) as p:
    #         max_ = len(audio_files)
    #         for _ in tqdm(p.imap_unordered(partial(extract_audio_feat, feature_extractor=feature_extractor, device=device, args=args), audio_files), total=max_):
    #             pass