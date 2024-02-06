import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from pathlib import Path
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from benchmark.tasks.MEDLEYDB.data_splits import DATA_SPLIT


class MEDLEYDBAudioDataset(data.Dataset):

    # Constant taken from https://github.com/drwangxian/dilated_conv_model_for_melody_extraction
    note_min = 23.6
    note_range = torch.arange(360) / 5.0
    note_range = note_range + note_min
    note_range = note_range.to(torch.float32)

    def __init__(self, cfg, split="train"):
        dataset_cfg = cfg.dataset
        pretrain_cfg = cfg.model.feature_extractor.pretrain
        audio_dir = dataset_cfg.input_dir
        self.sample_rate = pretrain_cfg.target_sr
        metadata_dir = dataset_cfg.metadata_dir
        self.clip_duration = getattr(
            dataset_cfg.audio_loader.crop_to_length_in_sec, split
        )
        assert audio_dir is not None, "Please specify audio directory"
        assert metadata_dir is not None, "Please specify label directory"
        audio_dir = Path(audio_dir)
        metadata_dir = Path(metadata_dir)
        track_names = DATA_SPLIT[split]

        label_files = [
            metadata_dir / f"{track_name}_MELODY2.csv" for track_name in track_names
        ]
        audio_files = [
            audio_dir / f"{track_name}/{track_name}_MIX.wav"
            for track_name in track_names
        ]
        orig_sample_rate = torchaudio.info(audio_files[0]).sample_rate

        self.table = []
        for idx, label_file in tqdm(enumerate(label_files), total=len(label_files)):
            times_labels = torch.Tensor(np.genfromtxt(label_file, delimiter=","))
            label_notes = MEDLEYDBAudioDataset.hz_to_midi_fn(times_labels[:, 1]).view(
                -1, 1
            )
            times_labels = torch.hstack((times_labels, label_notes))
            time_offsets = torch.arange(
                0, times_labels[-1, 0] + self.clip_duration, self.clip_duration
            )
            intervals = torch.vstack((time_offsets[:-1], time_offsets[1:]))
            label_invervals = torch.logical_and(
                times_labels[:, :1] > intervals[0], times_labels[:, :1] < intervals[1]
            ).T
            for offset, label_interval in zip(time_offsets, label_invervals):
                self.table.append(
                    [idx, (offset) * orig_sample_rate, times_labels[label_interval]]
                )
        self.audios = audio_files
        self.labels = label_files
        self.clip_frames_orig_sr = int((self.clip_duration) * orig_sample_rate)
        self.clip_frames_new_sr = int((self.clip_duration) * self.sample_rate)
        self.split = split

    def __len__(self):
        """
        Returns the length of the table.
        """
        return len(self.table)

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Parameters:
            index (int): The index of the item to retrieve.

        Returns:
            clip (torch.Tensor): The audio clip.
            label (tuple): Time, frequency, and note
        """

        file_idx, offset, label = self.table[index]
        file_idx, offset = int(file_idx), int(offset)
        sr = torchaudio.info(self.audios[file_idx]).sample_rate
        actual_frames = self.clip_frames_orig_sr + min(offset, 0)
        clip, sr = torchaudio.load(
            self.audios[file_idx], frame_offset=max(offset, 0), num_frames=actual_frames
        )
        clip_rs = torchaudio.functional.resample(clip.mean(dim=0), sr, self.sample_rate)
        if self.split == "train":
            pad_size = self.clip_frames_new_sr - clip_rs.shape[-1]
            pad_shape = (pad_size * (offset <= 0), pad_size * (offset > 0))
            clip = F.pad(clip_rs, pad_shape)
        else:
            clip = clip_rs
        return clip, offset / sr, label

    @staticmethod
    def collate_fn(batch):
        clips, offsets, labels = zip(*batch)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        return torch.stack(clips), torch.Tensor(offsets), labels

    @staticmethod
    def hz_to_midi_fn(freqs, ref_freq=440):

        notes = torch.zeros_like(freqs, dtype=torch.float32, device=freqs.device)
        positives = torch.nonzero(freqs)
        notes[positives] = 12 * torch.log2(freqs[positives] / ref_freq) + 69
        return notes
