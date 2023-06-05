import os

import json
import numpy as np
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark.utils.audio_utils import load_audio
from transformers import Wav2Vec2FeatureExtractor

class FeatureDataset(data.Dataset):  # TODO: need to finish this part
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False, split_version=0, low_quality_source=True):  # TODO: incoporate split_version into configuration. Change low quality to high quality source audios.
        if split == 'valid':
            split = 'validation'

        # TODO: this path has to be input from the config file
        self.feature_dir = feature_dir
        self.metadata_dir = os.path.join(metadata_dir, f'data/splits/split-{split_version}/autotagging_moodtheme-{split}.tsv')
        
        self.split_version = split_version
        self.low_quality_source = low_quality_source
        self.metadata = open(self.metadata_dir, 'r').readlines()[1:]

        self.all_paths = [line.split('\t')[3] for line in self.metadata]
        self.all_tags = [line.split('\t')[5:] for line in self.metadata]

        assert len(self.all_paths) == len(self.all_tags) == len(self.metadata)
        # read class2id
        self.class2id = self.read_class2id(metadata_dir, split_version)
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.layer = layer
        self.return_audio_path = return_audio_path

    def __getitem__(self, index):
        audio_path = self.all_paths[index]
        if self.low_quality_source:
            audio_path = audio_path.replace('.mp3', '.low.mp3')

        class_name = self.all_tags[index]
        features = np.load(os.path.join(self.feature_dir, audio_path + '.npy'))
        label = torch.zeros(len(self.class2id))  # TODO: how to deal with this?
        for c in class_name:
            label[self.class2id[c.strip()]] = 1

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
        output_labels = torch.vstack(output_labels)  # [Time, Label]

        if self.return_audio_path:
            return output_features, output_labels, audio_path
        return output_features, output_labels

    def __len__(self):
        return len(self.metadata)
    
    def read_class2id(self, metadata_dir, split_version):
        class2id = {}
        for split in ['train', 'validation', 'test']:
            data = open(os.path.join(metadata_dir, f'data/splits/split-{split_version}/autotagging_moodtheme-{split}.tsv'), "r").readlines()
            for example in data[1:]:
                tags = example.split('\t')[5:]
                for tag in tags:
                    tag = tag.strip()
                    if tag not in class2id:
                        class2id[tag] = len(class2id)
        return class2id
    
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
    