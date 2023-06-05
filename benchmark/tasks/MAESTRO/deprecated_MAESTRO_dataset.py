import os
import sys
import time
import collections
import logging
import math

import numpy as np
import h5py
import csv
import librosa
import sox
import pickle
import torchaudio
import torch
import numba as nb
from scipy.sparse import csr_matrix

from .utils import (float32_to_int16, int16_to_float32, traverse_folder)
from benchmark.tasks.MAESTRO.preprocess import TargetProcessor


class MaestroDataset(object):
    def __init__(self, hdf5s_dir, target_dir, segment_seconds=10, frames_per_second=75, 
        max_note_shift=0, target_sr=24000):
        """This class takes the meta of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
        """
        self.target_dir = target_dir
        self.hdf5s_dir = hdf5s_dir
        self.frames_per_second = frames_per_second
        self.target_sr = target_sr
        self.segment_seconds = segment_seconds
        self.segment_samples = None if segment_seconds is None else \
                                int(self.target_sr * self.segment_seconds)
        self.max_note_shift = max_note_shift
        self.begin_note = 21
        self.classes_num = 88

    def __getitem__(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}

        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num), 
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)

        note_shift = np.random.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:

            start_sample = 0
            end_sample = None
            start_frame = 0
            end_frame = None

            if self.segment_samples is not None:
                start_sample = int(start_time * self.target_sr)
                end_sample = start_sample + self.segment_samples
                start_frame = int(start_time * self.frames_per_second)
                end_frame = start_frame + int(self.segment_seconds * self.frames_per_second)

                if end_sample >= hf['waveform'].shape[0]:
                    start_sample -= self.segment_samples
                    end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample : end_sample])

            if note_shift != 0:
                """Augment pitch"""
                waveform = librosa.effects.pitch_shift(waveform, self.sample_rate, 
                    note_shift, bins_per_octave=12)
            
            waveform = torch.from_numpy(waveform)

            path_prefix = f'{self.target_dir}/{year}.{hdf5_name}'

            frame_roll = self.load_roll(f'{path_prefix}.target.frame_roll', 
                                        start_frame, end_frame)

            onset_roll = self.load_roll(f'{path_prefix}.target.onset_roll', 
                                        start_frame, end_frame)
            
            offset_roll = self.load_roll(f'{path_prefix}.target.offset_roll', 
                                        start_frame, end_frame)

        return waveform, frame_roll # (frame_roll, onset_roll, offset_roll)
    
    def load_roll(self, path, start_frame, end_frame):
        with open(path, 'rb') as f:
            roll = csr_matrix.todense(pickle.load(f))[start_frame : end_frame, :]
            roll = torch.from_numpy(roll).float()
        return roll


class Sampler(object):
    def __init__(self, hdf5s_dir, split, 
            batch_size, segment_seconds=10, hop_seconds=1, epoch_size=-1, shuffle=False, drop_last=False):
        """Sampler is used to sample segments for training or evaluation.

        Args:
          hdf5s_dir: str
          split: 'train' | 'validation' | 'test'
          segment_seconds: float
          hop_seconds: float
          batch_size: int
        """
        assert split in ['train', 'validation', 'test']
        self.hdf5s_dir = hdf5s_dir
        self.segment_seconds = segment_seconds
        self.hop_seconds = hop_seconds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        (hdf5_names, hdf5_paths) = traverse_folder(hdf5s_dir)
        self.segment_list = []

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    self.speed_append(self.segment_list, year, audio_name, hf.attrs['duration'])
                    
        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        logging.info('{} segments: {}'.format(split, len(self.segment_list)))

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        if self.shuffle: np.random.shuffle(self.segment_indexes)
        if epoch_size == -1: 
            self.epoch_size = len(self.segment_list)
        else:
            self.epoch_size = epoch_size

    def __iter__(self):
        batch_segment_list = []
        for idx in self.segment_indexes:
            batch_segment_list.append(self.segment_list[idx])
            self.pointer += 1

            if len(batch_segment_list) == self.batch_size:
                yield batch_segment_list
                batch_segment_list = []

            if self.pointer >= self.epoch_size:
                self.pointer = 0
                if self.shuffle: np.random.shuffle(self.segment_indexes)

        if len(batch_segment_list) > 0 and not self.drop_last:
            yield batch_segment_list
            
    def __len__(self):
        if self.drop_last:
            return self.epoch_size // self.batch_size
        else:
            return math.ceil(self.epoch_size / self.batch_size)
    
    @nb.jit(forceobj=True)
    def speed_append(self, segment_list, year, audio_name, duration):
        start_time = 0.0
        while (start_time + self.segment_seconds < duration):
            self.segment_list.append([year, audio_name, start_time])
            start_time += self.hop_seconds

