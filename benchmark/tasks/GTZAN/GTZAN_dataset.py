import os

import numpy as np
from torch.utils import data
import pandas as pd
import torch
import torch.nn.functional as F

from benchmark.utils.audio_utils import load_audio, chunk_audio
from transformers import Wav2Vec2FeatureExtractor

class FeatureDataset(data.Dataset):
    def __init__(self, feature_dir, metadata_dir, split, layer, return_audio_path=False):
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_filtered.txt'), 
                                    names = ['audio_path'])
        self.feature_dir = feature_dir
        self.class2id = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.layer = layer
        self.return_audio_path = return_audio_path

    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        feature_path = os.path.join(self.feature_dir, audio_path + '.npy')

        features = np.load(feature_path)
        label = self.class2id[audio_path.split('/')[0]]

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
        features = torch.vstack(features).squeeze(1) # [batch * Time, Feature]
        if len(labels[0].shape) == 1:
            labels = torch.cat(labels) # [batch * Time,], Time=1
        else:
            labels = torch.vstack(labels) # [batch * Time,]
        # print(features.shape, labels.shape)        
        return features, labels
        # return features, labels.view(-1)
    
    @staticmethod
    def test_collate_fn(batch):
        features = []
        labels = []
        index = []
        for i, example in enumerate(batch):
            # print(example)
            # print(example[0].shape, example[1][0], [i]*len(example[1]))
            features.append(example[0]) # [n_chunk, 1, Feature]
            labels.append(example[1][0]) # a single tensor, taken from the list example[1], where len(example[1])=n_chunk
            index.append([i] * len(example[1])) # [i, i, i ... i], use same index for all the chunks of one audio sample in a batch
        features = torch.vstack(features).squeeze(1) # [batch * n_chunk, Feature]
        labels = torch.vstack(labels).squeeze(1) # [batch,]
        index = torch.tensor(sum(index, [])) # [batch * n_chunk,]
        # print(features.shape, labels.shape, index.shape)
        return features, labels, index

class AudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, 
                 sample_duration=None, return_audio_path=False, 
                 sliding_window_size_in_sec=None, sliding_window_overlap_in_percent=0.0, 
                 target_sr=24000, wav_layernorm=False):
        # self.cfg = cfg
        self.metadata = pd.read_csv(filepath_or_buffer=os.path.join(metadata_dir, f'{split}_filtered.txt'), 
                                    names = ['audio_path'])
        self.split = split
        
        self.audio_dir = audio_dir
        self.class2id = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
        self.id2class = {v: k for k, v in self.class2id.items()}
        self.return_audio_path = return_audio_path
        self.sample_rate = target_sr # 16000, 24000
        self.sliding_window_size_in_sec = sliding_window_size_in_sec
        self.sliding_window_overlap_in_percent = sliding_window_overlap_in_percent
        self.sample_duration = sample_duration * self.sample_rate if sample_duration else None
        # extractor
        # set the sample rate same as the audio model trained with.
        # the audios in downstream dataset would be automatically converted to the same
        print(f'debug: init with {audio_dir, metadata_dir, split, sample_duration, return_audio_path, sliding_window_size_in_sec, sliding_window_overlap_in_percent, target_sr, wav_layernorm}')
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sample_rate,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=wav_layernorm,
        )
    

    def process_wav(self, waveform):
        # return the same shape
        return self.processor(
            waveform,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True).input_values[0]
    
    # def __getitem__(self, index):
    #     if self.split == 'train':
    #         return self.__getitem__train(index)
    #     else:
    #         return self.__getitem__test(index)
        
    # def __getitem__train(self, index):
    #     audio_path = self.metadata.iloc[index][0]
    #     audio = load_audio(os.path.join(self.audio_dir, audio_path), 
    #         target_sr = self.sample_rate,
    #         is_mono = True,
    #         is_normalize =  True,
    #     )

    #     # sample a duration of audio from random start
    #     if self.sample_duration is not None:  
    #         # if audio is shorter than sample_duration, pad it with zeros
    #         if audio.shape[1] <= self.sample_duration:  
    #             audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
    #         else:
    #             random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
    #             audio = audio[:, random_start:random_start+self.sample_duration]
        
    #     # preprocess and reshaping
    #     audio_features = self.process_wav(audio)

    #     # # convert
    #     # audio_features = self.processor(audio, return_tensors="pt", sampling_rate=self.cfg.target_sr, padding=True).input_values[0]
        
    #     label = self.class2id[audio_path.split('/')[0]]
    #     if self.return_audio_path:
    #         return audio_features, label, audio_path
    #     return audio_features, label
    
    def __getitem__(self, index):
        audio_path = self.metadata.iloc[index][0]
        audio = load_audio(os.path.join(self.audio_dir, audio_path), 
            target_sr = self.sample_rate,
            is_mono = True,
            is_normalize =  True,
        )


        # chunk and ensemble labels
        if self.sliding_window_size_in_sec:
            audios = chunk_audio(wav=audio, sample_rate=self.sample_rate,
                               sliding_window_size_in_sec=self.sliding_window_size_in_sec, sliding_window_overlap_in_percent=self.sliding_window_overlap_in_percent,
                               last_chunk_align='overlap')
        else:
            # sample a duration of audio from random start
            if self.sample_duration is not None:  
                # if audio is shorter than sample_duration, pad it with zeros
                if audio.shape[1] <= self.sample_duration:  
                    audio = F.pad(audio, (0, self.sample_duration - audio.shape[1]), 'constant', 0)
                else:
                    random_start = np.random.randint(0, audio.shape[1] - self.sample_duration)
                    audio = audio[:, random_start:random_start+self.sample_duration]
                audios = [audio]
            
            # features = feature_extractor(wav, layer=args.layer, reduction=args.reduction)

        label = self.class2id[audio_path.split('/')[0]]

        # audio_chunks =[]
        labels = []
        # preprocess and reshaping
        for i, audio in enumerate(audios):
            audios[i] = self.process_wav(audio) # a list
            labels.append(label)

        audios = torch.stack(audios, dim=0)  # [Time, Feature]
        labels = np.stack(labels, axis=0)  # [Time, Label]
        labels = torch.from_numpy(labels).long()

        if self.return_audio_path:
            return audios, labels, audio_path
        return audios, labels

    def __len__(self):
        return len(self.metadata)


    @staticmethod
    def train_collate_fn(batch):
        features, labels = zip(*batch)
        # print('auido shape', features[0].shape)
        features = torch.vstack(features).squeeze(1)
        # print('auido shape', features.shape)
        if len(labels[0].shape) == 1:
            labels = torch.cat(labels)
        else:
            labels = torch.vstack(labels)
        # print(features.shape, labels.shape)
        return features, labels
        
    
    @staticmethod
    def test_collate_fn(batch):
        features = []
        labels = []
        index = []
        for i, example in enumerate(batch):
            # print(example)
            features.append(example[0]) # a list of audio chunk(s)
            labels.append(example[1][0]) # label as an int
            index.append([i] * len(example[1]))
        features = torch.vstack(features).squeeze(1)
        labels = torch.vstack(labels).squeeze(1) #
        index = torch.tensor(sum(index, []))
        # print(features.shape, labels.shape, index.shape)
        return features, labels, index
