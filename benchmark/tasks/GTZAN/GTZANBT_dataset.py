"""
dataset
feature  75Hz 768 dim 没办法存 虽然 guitarset 应该可以 / audio
preprocess label
"""
import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F
import jams
import torchaudio

from benchmark.utils.audio_utils import load_audio
from transformers import Wav2Vec2FeatureExtractor

class BCEAudioDataset(data.Dataset):
    def __init__(self, cfg, split="train", return_audio_path=False):
        self.cfg = cfg
        self.audio_dir = cfg.dataset.input_dir
        self.metadata_dir = cfg.dataset.metadata_dir
        self.split = split
        self.datalist = pd.read_csv(
            filepath_or_buffer=f"{self.audio_dir}/../{split}_filtered.txt", 
            names = ['audio_path']
        )

        self.class2id = {
            'other': 0, 
            'beat': 1, 
            # 'downbeat': 2,
        }

        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path

        # assuming transformer models
        self.pretrain_cfg = cfg.model.feature_extractor.pretrain
        self.target_sr = self.pretrain_cfg.target_sr
        self.label_freq = self.pretrain_cfg.token_rate

        if self.pretrain_cfg.n_tranformer_layer:
            self.is_transformer = True
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.pretrain_cfg.huggingface_model_name,
                trust_remote_code=True,
            )
        else:
            self.is_transformer = False
            self.processor = None

        sample_duration = getattr(cfg.dataset.audio_loader.crop_to_length_in_sec, split)
        self.sample_duration = int(sample_duration * self.target_sr)
        self.crop_randomly = getattr(cfg.dataset.audio_loader.crop_randomly, split)
        self.pad = getattr(cfg.dataset.audio_loader.pad, split)
        self.return_start = getattr(cfg.dataset.audio_loader.return_start, split)
        
    def process_wav(self, waveform):
        if self.is_transformer:
            waveform = self.processor(
                waveform,
                return_tensors="pt",
                sampling_rate=self.target_sr,
                padding=True
            ).input_values[0].squeeze(0)
        return waveform

    def __getitem__(self, index):
        audio_path = self.datalist.iloc[index][0]
        audio = load_audio(os.path.join(self.audio_dir, audio_path), 
            target_sr=self.target_sr,
            is_mono=self.cfg.dataset.audio_loader.is_mono,
            is_normalize=self.cfg.dataset.audio_loader.is_normalize, # TODO: check if this is necessary
        )


        # sample a duration of audio from random start
        if self.sample_duration is not None:
            # if audio is shorter than sample_duration, pad it with zeros
            # if self.split != "test":
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
                paded = True
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
                paded = None
        
        # preprocess and reshaping
        audio_features = self.process_wav(audio)
        
        file_jam = jams.load(f"{self.metadata_dir}/{audio_path.split('/')[-1]}.jams")
        info = torchaudio.info(os.path.join(self.audio_dir, audio_path))
        duration = info.num_frames / info.sample_rate
        label = np.zeros((int(self.label_freq *  max(duration, self.sample_duration / self.target_sr)) + 1, 2))
        for annotation in file_jam.search(namespace='beat'):
            if annotation["sandbox"]["annotation_type"] == "beat":
                label_idx = [round(self.label_freq * i.time) for i in annotation["data"]]
                label_idx = np.array([idx for idx in label_idx if idx < label.shape[0] ])  # incase gtzan_reggae_00002.jams
                label[label_idx, 0] = 1
                if self.split == "train":
                    label[label_idx[1:]-1, 0] = .95
                    label[label_idx[1:]-2, 0] = .5
                    label[label_idx[:-1]+1, 0] = .95
                    label[label_idx[:-1]+2, 0] = .5
                
            elif annotation["sandbox"]["annotation_type"] == "downbeat":
                label_idx = [round(self.label_freq * i.time) for i in annotation["data"]]
                label_idx = np.array(label_idx)  # incase gtzan_reggae_00002.jams
                # try:
                if len(label_idx):
                    label[label_idx, 1] = 1
                    if self.split == "train":
                        label[label_idx[1:]-1, 1] = .95
                        label[label_idx[1:]-2, 1] = .5
                        label[label_idx[1:]-3, 1] = .25
                        label[label_idx[:-1]+1, 1] = .95
                        label[label_idx[:-1]+2, 1] = .5 
                        label[label_idx[:-1]+3, 1] = .25
                # except:  #train:jazz.00018.wav jazz.00020.wav jazz.00014.wav;  valid:jazz.00009.wav jazz.00010.wav
                #     print(audio_path, len(label_idx))
        if not paded:  
            label_start = int(random_start * self.label_freq / self.target_sr)
            label = label[label_start: int(self.sample_duration / self.target_sr * self.label_freq + label_start) + 1, :]
        if self.return_audio_path:
            return audio_features, label, audio_path
        # if label.shape[0] < 2251:
        #     print(audio_path, audio_features.shape, label.shape)
        # set the following torch to float32
        return audio_features, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.datalist)

# multi-task for beat and downbeat with multiple crossentropy loss
class AudioDataset(BCEAudioDataset):
    def __getitem__(self, index):
        audio_path = self.datalist.iloc[index][0]
        audio = load_audio(os.path.join(self.audio_dir, audio_path), 
            target_sr = self.target_sr,
            is_mono = True,
            is_normalize =  True,
        )

        # sample a duration of audio from random start
        # paded = None
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
                paded = True
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
                paded = None
        
        # preprocess and reshaping
        audio_features = self.process_wav(audio)

        file_jam = jams.load(f"{self.metadata_dir}/{audio_path.split('/')[-1]}.jams")
        info = torchaudio.info(os.path.join(self.audio_dir, audio_path))
        duration = info.num_frames / info.sample_rate
        label = np.zeros((int(self.label_freq * max( duration, self.sample_duration / self.target_sr)) + 1))
        for annotation in file_jam.search(namespace='beat'):
            if annotation["sandbox"]["annotation_type"] == "beat":
                label_idx = [round(self.label_freq * i.time) for i in annotation["data"]]
                label_idx = np.array([idx for idx in label_idx if idx < label.shape[0] ])  # incase gtzan_reggae_00002.jams
                label[label_idx] = 1
                if self.split == "train":
                    label[label_idx[1:]-1] = 1
                    label[label_idx[:-1]+1] = 1
                
            # elif annotation["sandbox"]["annotation_type"] == "downbeat":

        if not paded:  
            label_start = int(random_start * self.label_freq / self.target_sr)
            label = label[label_start: int(self.sample_duration / self.target_sr * self.label_freq + label_start)+1]
        if self.return_audio_path:
            return audio_features, label, audio_path
        # if label.shape[0] < 2251:
        #     print(audio_path, audio_features.shape, label.shape)
        return audio_features, label

    