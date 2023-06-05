import os

import numpy as np
from torch.utils import data
import pandas as pd
from benchmark.utils.audio_utils import load_audio
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor
import torch

class FeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False):
        
        metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}.tsv'), 
                                    sep='\t',
                                    names = ['uuid', 'audio_path'])
        self.class_names = np.load(os.path.join(metadata_dir, 'tags.npy'))
        # use python list, should faster than pandas iloc
        self.uuids = []
        self.audio_paths = []
        for i in range(len(metadata)):
            uuid, audio_path = metadata.iloc[i]
            self.uuids.append(uuid)
            self.audio_paths.append(audio_path)

        self.feature_dir = feature_dir
        self.labels = np.load(os.path.join(metadata_dir, 'binary_label.npy'))
        self.layer = layer
        self.return_audio_path = return_audio_path

    def __getitem__(self, index):
        # uuid, audio_path = self.metadata.iloc[index]
        uuid = self.uuids[index]
        audio_path = self.audio_paths[index]

        feature_path = os.path.join(self.feature_dir, audio_path + '.npy')
        features = np.load(feature_path)
        label = self.labels[uuid]

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
        # output_labels to numpy
        output_labels = np.stack(output_labels, axis=0)  # [Time, Label]
        output_labels = torch.from_numpy(output_labels).float()

        if self.return_audio_path:
            return output_features, output_labels, audio_path
        return output_features, output_labels

    def __len__(self):
        return len(self.uuids)
    
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
        metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}.tsv'), 
                                    sep='\t',
                                    names = ['uuid', 'audio_path'])
        # use python list, should faster than pandas iloc
        self.uuids = []
        self.audio_paths = []
        for i in range(len(metadata)):
            uuid, audio_path = metadata.iloc[i]
            self.uuids.append(uuid)
            self.audio_paths.append(audio_path)

        self.audio_dir = audio_dir
        self.labels = np.load(os.path.join(metadata_dir, 'binary_label.npy'))
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
        # uuid, audio_path = self.metadata.iloc[index]
        uuid = self.uuids[index]
        audio_path = self.audio_paths[index]

        audio = load_audio(os.path.join(self.audio_dir, audio_path))
        # sample a duration of audio from random start
        if self.sample_duration is not None:  
            # if audio is shorter than sample_duration, pad it with zeros
            if audio.shape[1] <= self.sample_duration:  
                audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
            else:
                random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                audio = audio[:, random_start:random_start+self.sample_duration]
        
        # preprocess and reshaping
        audio = self.process_wav(audio)

        label = self.labels[uuid].astype(np.float16)
        if self.return_audio_path:
            return audio, label, audio_path
        return audio, label

    def __len__(self):
        return len(self.uuids)

if __name__ == '__main__':
    # only for testing
    feature_dir = "/home/yrb/code/MusicAudioPretrain/data/MTT/hubert_features/HF_model_HuBERT_base_MPD_train_1Kh_valid_300h_iter1_250k_vanilla_model_ncluster_500_feature_layer_all_reduce_mean"
    metadata_dir = "/home/yrb/code/MusicAudioPretrain/data/MTT"
    split = "train"
    layer = 'all'
    dataset = FeatureDataset(feature_dir, metadata_dir, split, layer)
    dataset[0]
