import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark.utils.audio_utils import load_audio
from transformers import Wav2Vec2FeatureExtractor

class FeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False):

        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_s.txt'), 
                                    names = ['audio_path'])
        self.feature_dir = feature_dir
        self.class2id = {'f1':0, 'f2':1, 'f3':2, 'f4':3, 'f5':4, 'f6':5, 'f7':6, 'f8':7, 'f9':8, 'm1':9, 'm2':10, 'm3':11, 'm4':12, 'm5':13, 'm6':14, 'm7':15, 'm8':16, 'm9':17, 'm10':18, 'm11':19}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.layer = layer
        self.return_audio_path = return_audio_path

    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        feature_path = os.path.join(self.feature_dir, audio_path + '.npy')
        features = np.load(feature_path)
        label = self.class2id[audio_path.split('/')[1].split('_')[0]]

        output_features = []
        output_labels = []
        for i in range(features.shape[1]):
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
        output_labels = np.stack(output_labels, axis=0)  # [Time, Label]
        output_labels = torch.from_numpy(output_labels).long()

        if self.return_audio_path:
            return output_features, output_labels, audio_path
        return output_features, output_labels

    def __len__(self):
        return len(self.metadata)
    
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
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_s.txt'), 
                                    names = ['audio_path'])
        self.audio_dir = audio_dir
        self.class2id = {'f1':0, 'f2':1, 'f3':2, 'f4':3, 'f5':4, 'f6':5, 'f7':6, 'f8':7, 'f9':8, 'm1':9, 'm2':10, 'm3':11, 'm4':12, 'm5':13, 'm6':14, 'm7':15, 'm8':16, 'm9':17, 'm10':18, 'm11':19}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = 16000
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
        # extractor
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
        audio_path = self.metadata.iloc[index][0]
        audio = load_audio(os.path.join(self.audio_dir, audio_path), 
            target_sr = self.sample_rate,
            is_mono = True,
            is_normalize =  True,
        )


        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
        
        # preprocess and reshaping
        audio_features = self.process_wav(audio)

        # # convert
        # audio_features = self.processor(audio, return_tensors="pt", sampling_rate=self.cfg.target_sr, padding=True).input_values[0]
        
        label = self.class2id[audio_path.split('/')[1].split('_')[0]]
        if self.return_audio_path:
            return audio_features, label, audio_path
        return audio_features, label

    def __len__(self):
        return len(self.metadata)


