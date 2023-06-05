import os

import numpy as np
from torch.utils import data
import pandas as pd
import json
import torch
import torch.nn.functional as F

from benchmark.utils.audio_utils import load_audio
from transformers import Wav2Vec2FeatureExtractor

class FeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False):
        """
        each song is a 45s clip"""
        
        self.metadata = os.path.join(metadata_dir, 'meta.json')
        with open(self.metadata) as f:
            self.metadata = json.load(f)
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.feature_dir = feature_dir
        self.classes = """arousal, valence""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.layer = layer
        self.return_audio_path = return_audio_path

    def __getitem__(self, index):
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'  # relative path to audio folder
        feature_path = os.path.join(self.feature_dir, audio_path + '.npy')
        features = np.load(feature_path)
        label = torch.from_numpy(np.array(self.metadata[audio_name_without_ext]['y'], dtype=np.float32))

        output_features = []
        output_labels = []
        for i in range(features.shape[1]): # [L, T, H]
            feature = features[:, i, :]
            if feature.ndim == 2: #  [Layer=13, 1, Feature=768]
                if self.layer == 'all':
                    feature = torch.from_numpy(feature[1:]).unsqueeze(dim=0)  # [Layer, Feature]
                else:
                    feature = torch.from_numpy(feature[int(self.layer)]).unsqueeze(dim=0)  # [Feature]
            else:
                raise ValueError
            output_features.append(feature)
            output_labels.append(label)
        output_features = torch.stack(output_features, dim=0)  # [Time, Feature]
        output_labels = torch.vstack(output_labels)  # [Time, Label]
        if self.return_audio_path:
            return output_features, output_labels, audio_path
        return output_features, output_labels

    def __len__(self):
        return len(self.audio_names_without_ext)
    
    @staticmethod
    def train_collate_fn(batch):
        features, labels = zip(*batch)
        features = torch.vstack(features).squeeze(1)
        if len(labels[0].shape) == 1:
            labels = torch.cat(labels)
        else:
            labels = torch.vstack(labels)
        return features, labels
        # return features, labels.view(-1)
    
    @staticmethod
    def test_collate_fn(batch):
        features = []
        labels = []
        index = []
        for i, example in enumerate(batch):
            features.append(example[0])
            labels.append(example[1][0])
            index.append([i] * len(example[1]))
        features = torch.vstack(features).squeeze(1)
        labels = torch.vstack(labels).squeeze(1)
        index = torch.tensor(sum(index, []))
        return features, labels, index

class AudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=None, return_audio_path=False):
        self.metadata = os.path.join(metadata_dir, 'meta.json')
        with open(self.metadata) as f:
            self.metadata = json.load(f)
        self.audio_dir = audio_dir
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.classes = """arousal, valence""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path

        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None

        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

    def process_wav(self, waveform):
        # return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]

    def __getitem__(self, index):
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'
        audio = load_audio(os.path.join(self.audio_dir, audio_path))
        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
        
        audio = self.process_wav(audio)
        # label = self.metadata[audio_name_without_ext]['y']
        label = np.array(self.metadata[audio_name_without_ext]['y'], dtype=np.float32)

        if self.return_audio_path:
            return audio, label, audio_path
        return audio, label

    def __len__(self):
        return len(self.audio_names_without_ext)


