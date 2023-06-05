"""MAESTRO dataset preprocessing script.

Adapted from:
https://github.com/bytedance/piano_transcription/blob/master/utils/features.py
https://github.com/bytedance/piano_transcription/blob/master/utils/utilities.py
https://github.com/bytedance/piano_transcription/blob/master/pytorch/calculate_score_for_paper.py

Adapted by: Ruibin Yuan
"""

import os, sys
import time
import logging
import csv
import pickle
import collections
import argparse

import numpy as np
import h5py
from tqdm import tqdm
import torch
from scipy.sparse import csr_matrix

# fix import error, add src to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

from benchmark.tasks.MAESTRO.utils import (create_folder, float32_to_int16, int16_to_float32, create_logging, 
    get_filename, read_midi, load_audio, traverse_folder)


def read_metadata(csv_path):
    """Read metadata of MAESTRO dataset from csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict, dict, e.g. {
        'canonical_composer': ['Alban Berg', ...], 
        'canonical_title': ['Sonata Op. 1', ...], 
        'split': ['train', ...], 
        'year': ['2018', ...]
        'midi_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.midi', ...], 
        'audio_filename': ['2018/MIDI-Unprocessed_Chamber3_MID--AUDIO_10_R3_2018_wav--1.wav', ...],
        'duration': [698.66116031, ...]}
    """

    with open(csv_path, 'r') as fr:
        reader = csv.reader(fr, delimiter=',')
        lines = list(reader)

    meta_dict = {'canonical_composer': [], 'canonical_title': [], 'split': [], 
        'year': [], 'midi_filename': [], 'audio_filename': [], 'duration': []}

    for n in range(1, len(lines)):
        meta_dict['canonical_composer'].append(lines[n][0])
        meta_dict['canonical_title'].append(lines[n][1])
        meta_dict['split'].append(lines[n][2])
        meta_dict['year'].append(lines[n][3])
        meta_dict['midi_filename'].append(lines[n][4])
        meta_dict['audio_filename'].append(lines[n][5])
        meta_dict['duration'].append(float(lines[n][6]))

    for key in meta_dict.keys():
        meta_dict[key] = np.array(meta_dict[key])
    
    return meta_dict


def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = './data/MAESTRO'

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v2.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'preprocess_logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in tqdm(range(audios_num)):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        audio = load_audio(audio_path, target_sr=args.target_sr, is_mono=True)
        audio = audio.squeeze().numpy()

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note, 
        classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events, 
        extend_pedal=True, note_shift=0):
        """Process MIDI events of an audio segment to target for training, 
        includes: 
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording, 
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is 
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
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

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35}, 
            {'onset_time': 150.54, 'offset_time': 152.06}, 
            ...]
        """

        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + self.segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}    # Used to store onset of notes to be paired with offsets
        pedal_dict = {}     # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for 
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149, 
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)  
        ex_bgn_idx = max(bgn_idx - _delta, 0)
        
        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i], 
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note, 
                            'onset_time': buffer_dict[midi_note]['onset_time'], 
                            'offset_time': midi_events_time[i], 
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'], 
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note, 
                'onset_time': buffer_dict[midi_note]['onset_time'], 
                'offset_time': start_time + self.segment_seconds, 
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'], 
                'offset_time': start_time + self.segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)
        
        # Prepare targets
        frames_num = int(round(self.segment_seconds * self.frames_per_second)) + 1
        onset_roll = np.zeros((frames_num, self.classes_num))
        offset_roll = np.zeros((frames_num, self.classes_num))
        reg_onset_roll = np.ones((frames_num, self.classes_num))
        reg_offset_roll = np.ones((frames_num, self.classes_num))
        frame_roll = np.zeros((frames_num, self.classes_num))
        velocity_roll = np.zeros((frames_num, self.classes_num))
        mask_roll = np.ones((frames_num, self.classes_num))
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num)
        pedal_offset_roll = np.zeros(frames_num)
        reg_pedal_onset_roll = np.ones(frames_num)
        reg_pedal_offset_roll = np.ones(frames_num)
        pedal_frame_roll = np.zeros(frames_num)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note) 
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0) : fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)
                
                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame :, piano_note] = 0     

        # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0) : fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll, 
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll, 
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll, 
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
            }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0     # Index of note events
        while pedal_events: # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}    # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal, 
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:
                    
                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx
                
                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by 
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)
        
        locts = np.where(input < 0.5)[0] 
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output


def save_sparse(dense, path, framework='scipy'):
    """Save a dense matrix as a scipy or torch sparse matrix.
    Scipy version is usually smaller than torch version."""
    if framework == 'torch':
        sparse = torch.from_numpy(dense).to_sparse()
        torch.save(sparse, path)
    elif framework == 'scipy':
        sparse = csr_matrix(dense)
        with open(path, 'wb') as f:
            pickle.dump(sparse, f)
    else:
        raise ValueError("Framework {} not supported.".format(framework))


def preprocess_label(args):
    HDF5S_DIR = "data/MAESTRO/hdf5s/maestro"
    TARGET_DIR = "data/MAESTRO/maestro_target" # TODO: replace to "data/MAESTRO/maestro_target"
    SEGMENT_SEC = None # "If None, use the whole audio file."
    FRAMES_PER_SEC = args.frames_per_second
    HOP_SEC = 1 
    TARGET_SR = args.target_sr
    BEGIN_NOTE = 21 # "The lowest note in the piano."
    CLASS_NUM = 88 # "The number of notes in the piano."
    EXTEND_PEDAL = True
    NOTE_SHIFT = 0
    RESUME = args.resume

    assert HOP_SEC > 0
    create_folder(TARGET_DIR)

    (hdf5_names, hdf5_paths) = traverse_folder(HDF5S_DIR)
    for hdf5_path in tqdm(hdf5_paths):
        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hdf5_path.split('/')[-1]
            year = hf.attrs['year'].decode()
            duration_meta = hf.attrs['duration']
            audio = int16_to_float32(hf['waveform'][:])
            duration = len(audio) / TARGET_SR
            assert abs(duration - duration_meta) / duration_meta < 0.1, \
                f"Duration mismatch: {duration} vs {duration_meta}, check sample rate of {hdf5_path}"
            cur_segement_seconds = duration if SEGMENT_SEC is None else SEGMENT_SEC

            if SEGMENT_SEC is None:
                save_prefix = f'{TARGET_DIR}/{year}.{audio_name}'
            else:
                save_prefix = f'{TARGET_DIR}/{year}.{audio_name}.{start_time}'
            
            if os.path.exists(save_prefix + '.pedal') and RESUME:
                continue

            target_processor = TargetProcessor(
                cur_segement_seconds, 
                FRAMES_PER_SEC, 
                BEGIN_NOTE, 
                CLASS_NUM
            )

            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]

            start_time = 0
            while (start_time + cur_segement_seconds <= duration):
                # Process MIDI events to target
                (target_dict, note_events, pedal_events) = \
                    target_processor.process(start_time, midi_events_time, 
                        midi_events, extend_pedal=EXTEND_PEDAL, note_shift=NOTE_SHIFT)

                # Save target rolls
                for roll_name, roll in target_dict.items():
                    if roll_name in {'onset_roll', 'offset_roll', 'frame_roll', 'pedal_onset_roll', 'pedal_offset_roll', 'pedal_frame_roll'}:
                        roll = roll.astype('bool')
                    if roll_name in {'velocity_roll'}:
                        roll = roll.astype('int16')
                    if roll_name in {'mask_roll'}: # large and not used
                        continue
                    if roll_name in {'reg_onset_roll', 'reg_offset_roll', 'reg_pedal_onset_roll', 'reg_pedal_offset_roll'}:
                        roll = roll.astype('float32')
                    target_path = f'{save_prefix}.target.{roll_name}'
                    save_sparse(roll, target_path, framework='scipy')

                # Save note events
                note_events_path = f'{save_prefix}.note'
                with open(note_events_path, 'wb') as f:
                    pickle.dump(note_events, f)
                
                # Save pedal events
                pedal_events_path = f'{save_prefix}.pedal'
                with open(pedal_events_path, 'wb') as f:
                    pickle.dump(pedal_events, f)

                start_time += HOP_SEC


def main(args):
    if args.mode == 'audio':
        pack_maestro_dataset_to_hdf5(args)
    elif args.mode == 'label':
        preprocess_label(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, help="The mode of preprocessing. Can be  ['audio', 'label'].")
    parser.add_argument("--target_sr", type=int, required=False, default=24000, help="The target sampling rate.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="The path to the dataset directory, the resulting folder when you unzip maestro-v2.0.0.zip")

    # these 2 only used in label mode
    parser.add_argument("--frames_per_second", type=int, required=False, default=75, help="The number of frames per second.")
    parser.add_argument("--resume", action='store_true', help="Whether to resume label preprocessing.")

    args = parser.parse_args()
    main(args)
