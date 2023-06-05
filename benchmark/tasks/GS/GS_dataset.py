import os

import numpy as np
from torch.utils import data
import pandas as pd
import json
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from benchmark.utils.audio_utils import load_audio

class FeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False):
        """
        each song is segmented into 30s clips"""
        
        self.metadata = os.path.join(metadata_dir, 'meta.json')
        with open(self.metadata) as f:
            self.metadata = json.load(f)
        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.feature_dir = feature_dir
        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.layer = layer
        self.return_audio_path = return_audio_path
        self.ensemble = True if split in ['test', 'valid'] else False

        # construct unique ids for test ensemble
        # test ensemble will sooon be deprecated. 
        # simply extract chunk based features and will automatically do ensemble
        # requires reprocess of GS dataset
        # use the original uid, original wav, and original metadata
        # need to regenerate splits with the jukemir artist stratified protocol
        self.meta_ids = [self.metadata[k]["clip"]["audio_uid"] for k in self.metadata.keys() if self.metadata[k]['split'] == split] # actual uid in string
        self.meta_id2idx = {uid:idx for idx, uid in enumerate(set(sorted(self.meta_ids)))} # sorted will not change the original list
        self.meta_indices = [self.meta_id2idx[uid] for uid in self.meta_ids] # uid-binding unique index
        assert len(self.meta_indices) == len(self.audio_names_without_ext), f"for {split} data, meta indices {len(self.meta_indices)}, clip number {len(self.audio_names_without_ext)}, not aligned"
            

    def __getitem__(self, index):
        audio_name_without_ext = self.audio_names_without_ext[index]
        audio_path = audio_name_without_ext + '.wav'  # relative path to audio folder
        feature_path = os.path.join(self.feature_dir, audio_path + '.npy')
        features = np.load(feature_path)
        label = self.class2id[self.metadata[audio_name_without_ext]['y']]

        meta_idx = self.meta_indices[index]
        class_in_str = self.id2class[label]

        output_features = []
        output_labels = []
        output_meta_indices = []
        output_classes_in_str = []

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
            output_meta_indices.append(meta_idx)
            output_classes_in_str.append(class_in_str)

        output_features = torch.stack(output_features, dim=0)  # [Time, Feature]
        output_labels = torch.tensor(output_labels)  # [Time, Label]
        output_meta_indices = torch.tensor(output_meta_indices)  # [Time, Meta_idx]
        output_classes_in_str = np.array(output_classes_in_str)  # [Time, Class_in_str]

        if self.return_audio_path:
            return output_features, output_labels, audio_path

        if self.ensemble:
            return output_features, output_labels, output_meta_indices, output_classes_in_str
        else:
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
    
    @staticmethod
    def test_collate_fn(batch):
        features = []
        labels = []
        meta_indices = []
        classes_in_str = []
        index = []
        for i, example in enumerate(batch):
            features.append(example[0])
            labels.append(example[1][0])
            meta_indices.append(example[2][0])
            classes_in_str.append(example[3][0])
            index.append([i] * len(example[1]))
        features = torch.vstack(features).squeeze(1)
        labels = torch.vstack(labels).squeeze(1)
        meta_indices = torch.vstack(meta_indices).squeeze(1)
        classes_in_str = np.vstack(classes_in_str).squeeze(1)
        index = torch.tensor(sum(index, []))
        return features, labels, meta_indices, classes_in_str, index


class AudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sample_duration=None, return_audio_path=False):
        """
        each song is segmented into 30s clips"""

        self.metadata = os.path.join(metadata_dir, 'meta.json')
        self.ensemble = True if split in ['test', 'valid'] else False
        with open(self.metadata) as f:
            self.metadata = json.load(f)

        self.audio_names_without_ext = [k for k in self.metadata.keys() if self.metadata[k]['split'] == split]
        self.audio_dir = audio_dir

        self.classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        self.class2id = {c: i for i, c in enumerate(self.classes)}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        


        # construct unique ids for test ensemble
        if self.ensemble:
            self.meta_ids = [self.metadata[k]["clip"]["audio_uid"] for k in self.metadata.keys() if self.metadata[k]['split'] == split] # actual uid in string
            self.meta_id2idx = {uid:idx for idx, uid in enumerate(set(sorted(self.meta_ids)))} # sorted will not change the original list
            self.meta_indices = [self.meta_id2idx[uid] for uid in self.meta_ids] # uid-binding unique index
            assert len(self.meta_indices) == len(self.audio_names_without_ext), f"for {split} data, meta indices {len(self.meta_indices)}, clip number {len(self.audio_names_without_ext)}, not aligned"
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
        audio_name_without_ext =  self.audio_names_without_ext[index]
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
        # preprocess and reshaping
        audio = self.process_wav(audio)
        
        label = self.class2id[self.metadata[audio_name_without_ext]['y']]

        if self.return_audio_path:
            return audio, label, audio_path
            # return_elements.append(audio_path)

        if self.ensemble:
            # unique index regards to the song-level id + class information in string
            return audio, label, self.meta_indices[index], self.id2class[label]
        else:
            return audio, label

    def __len__(self):
        return len(self.audio_names_without_ext)


