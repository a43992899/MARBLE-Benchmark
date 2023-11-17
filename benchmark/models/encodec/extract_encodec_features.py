import os
import argparse
import typing as tp

import wget
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from encodec import EncodecModel
from encodec.utils import convert_audio

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
    return args

class EncodecFeature(nn.Module):
    def __init__(
            self,
            sr=24000,
            target_bandwidth=6.0,
        ) -> None:
        super().__init__()
        if sr == 24000:
            self.model = EncodecModel.encodec_model_24khz()
        elif sr == 48000:
            self.model = EncodecModel.encodec_model_48khz()
        else:
            raise NotImplementedError(f'sr={sr} is not supported')
        self.model.set_target_bandwidth(target_bandwidth)
    
    def get_audio_seg_embedding(self, frame_x: torch.Tensor):
        length = frame_x.shape[-1]
        duration = length / self.model.sample_rate
        assert self.model.segment is None or duration <= 1e-5 + self.model.segment
        
        if self.model.normalize:
            mono = frame_x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            frame_x = frame_x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
            
        emb = self.model.encoder(frame_x)
        return emb

    def get_audio_encodec_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.dim() == 3
        _, channels, length = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.model.segment_length
        # print(model.segment_length) # None，所以返回的就是所有的 dimention
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.model.segment_stride  # type: ignore
            assert stride is not None

        encoded_frame_embedings: tp.List[torch.Tensor] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            print(frame.shape)
            frame_embedding = get_audio_seg_embedding(self.model, frame)
            encoded_frame_embedings.append(frame_embedding)


        # return mean pooling according to time dimension
        if len(encoded_frame_embedings) == 1:
            # print(encoded_frame_embedings[0].shape) # [1, 128, 2251]
            return torch.mean(encoded_frame_embedings[0], dim=-1) # [1, 128] or # [B, 128]
        else:
            # print(torch.stack(encoded_frame_embedings).shape) # [N, 1, 128, 2251 (n samples)]
            return torch.mean(torch.cat(encoded_frame_embedings, dim=-1), dim=-1) # [B, 128]

    def forward(self, input_values):
        pass
        return




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

    device = 'cuda'
    audio_path = './data/GTZAN/genres/blues/blues.00000.wav'

    feature_extractor = EncodecFeature(
        args.
    )

    feature_extractor.to(device)
    
    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0) # [1, 1, n_samples], 第一个维度是 batch_size
    wav = wav.to(device)


    # Extract discrete codes from EnCodec
    with torch.no_grad():
        wav_encodec_embs = get_audio_encodec_embeddings(model, wav) # model.encode(wav)

    print(wav_encodec_embs.shape)