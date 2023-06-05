import os
import sys
import time
import collections
import logging
import math

import numpy as np
from torch.utils import data
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


########### 
# TODO: support hf preprocessor in dataset to avoid moving data multiple times between devices
# TODO: support return onset and offset rolls
# TODO: different batching strategy with test
# ########### 

class MaestroAudioDataset(data.Dataset):
    def __init__(self, audio_dir, metadata_dir, split, sliding_window_size_in_sec=10, hop_seconds=1, frames_per_second=75, 
        max_note_shift=0, target_sr=24000):
        """This class takes the idx of an audio segment as input, and return 
        the waveform and targets of the audio segment. This class is used by 
        DataLoader. 

        Note that bytedance transcription system uses a hop_seconds of 5s during evaluation.
        
        Args:
          feature_hdf5s_dir: str
          segment_seconds: float
          frames_per_second: int
          max_note_shift: int, number of semitone for pitch augmentation
        """
        self.target_dir = metadata_dir
        self.hdf5s_dir = audio_dir
        self.frames_per_second = frames_per_second
        self.target_sr = target_sr
        self.split = split
        self.segment_seconds = sliding_window_size_in_sec
        self.segment_samples = None if sliding_window_size_in_sec is None else \
                                int(self.target_sr * self.segment_seconds)
        self.max_note_shift = max_note_shift
        self.hop_seconds = hop_seconds
        self.begin_note = 21
        self.classes_num = 88

        self.init_meta()

    def __getitem__(self, idx):
        segment_meta = self.metadata[idx]
        return self.getitem_by_meta(segment_meta)
    
    def __len__(self):
        return len(self.metadata)

    def getitem_by_meta(self, meta):
        """Prepare input and target of a segment for training.
        
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 0.0}

        Returns:
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
            'pedal_frame_roll': (frames_num,)
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)

        note_shift = np.random.randint(low=-self.max_note_shift, 
            high=self.max_note_shift + 1)

        # Load hdf5
        with h5py.File(hdf5_path, 'r') as hf:
            # start_sample = 0
            # end_sample = None
            # start_frame = 0
            # end_frame = None

            # if self.split not in 'test':
            #     """Sample a segment from the audio during training,
            #     but use the whole audio during valid and test.
            #     """
            #     start_sample = int(start_time * self.target_sr)
            #     end_sample = start_sample + self.segment_samples
            #     start_frame = int(start_time * self.frames_per_second)
            #     end_frame = start_frame + int(self.segment_seconds * self.frames_per_second)

            #     if end_sample >= hf['waveform'].shape[0]:
            #         start_sample -= self.segment_samples
            #         end_sample -= self.segment_samples
            
            """Sample a segment from the audio during training,
            but use the whole audio during valid and test.
            """
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

    @nb.jit(forceobj=True)
    def speed_append(self, segment_list, year, audio_name, duration):
        start_time = 0.0
        while (start_time + self.segment_seconds < duration):
            self.metadata.append([year, audio_name, start_time])
            start_time += self.hop_seconds
    
    def init_meta(self):
        """Init meta of all segments.
        """

        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""

        (hdf5_names, hdf5_paths) = traverse_folder(self.hdf5s_dir)
        split_mapper = {'train': 'train', 'valid': 'validation', 'test': 'test'}
        self.metadata = []
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split_mapper[self.split]:
                    audio_name = hdf5_path.split('/')[-1]
                    year = hf.attrs['year'].decode()
                    self.speed_append(self.metadata, year, audio_name, hf.attrs['duration'])
    