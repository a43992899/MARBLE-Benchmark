import os, random, logging

import torch
import torchaudio
import numpy as np
from mido import MidiFile
np.set_printoptions(precision=4, suppress=True)


def load_audio(file_path, target_sr=24000, target_depth=16,is_mono=True, is_normalize=False,
                    crop_to_length_in_sec=None, crop_to_length_in_sample_points=None, crop_randomly=False, device=torch.device('cpu')):
    # TODO: deal with target_depth
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if is_normalize:
        waveform = waveform / waveform.abs().max()
    
    assert crop_to_length_in_sec is None or crop_to_length_in_sample_points is None, \
    "Only one of crop_to_length_in_sec and crop_to_length_in_sample_points can be specified"

    crop_duration_in_sample = None
    if crop_to_length_in_sec:
        crop_duration_in_sample = int(target_sr * crop_to_length_in_sec)
    elif crop_to_length_in_sample_points:
        crop_duration_in_sample = crop_to_length_in_sample_points

    if crop_duration_in_sample:
        if waveform.shape[-1] > crop_duration_in_sample:
            if crop_randomly:
                start = random.randint(0, waveform.shape[-1] - crop_duration_in_sample)
            else:
                start = 0
            waveform = waveform[..., start:start + crop_duration_in_sample]
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = waveform.to(device)
        resampler = resampler.to(device)
        waveform = resampler(waveform)
        
    return waveform


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0', 
            'control_change channel=0 control=64 value=127 time=0', 
            'control_change channel=0 control=64 value=63 time=236', 
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list), 
        'midi_event_time': np.array(time_in_second)}

    return midi_dict


def traverse_folder(folder):
    paths = []
    names = []
    
    for root, dirs, files in os.walk(folder):
        for name in files:
            filepath = os.path.join(root, name)
            names.append(name)
            paths.append(filepath)
            
    return names, paths


def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.
    return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)


def note_to_freq(piano_note):
    return 2 ** ((piano_note - 39) / 12) * 440


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


