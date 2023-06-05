import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio
import tqdm


def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):
    """Load audio file

    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root: Union[Path, str],
        sample_rate: float,
        seq_duration: Optional[float] = None,
        source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""



class FixedSourcesTrackFolderDataset(UnmixDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_file: str = "vocals.wav",
        mixture_file: str = "mixture.wav",
        interferer_files: List[str] = ["bass.wav", "drums.wav", "other.wav"],
        seq_duration: Optional[float] = None,
        samples_per_track: int = 64,
        random_chunks: bool = False,
        random_track_mix: bool = False,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        sample_rate: float = 44100.0,
        seed: int = 42,
    ) -> None:
        """A dataset that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.

        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.

        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.

        Example
        =======
        train/1/vocals.wav ---------------\
        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/

        train/1/vocals.wav -------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.samples_per_track = samples_per_track
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.mixture_file = mixture_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.resampler = torchaudio.transforms.Resample(44100, 24000)
        self.seed = seed
        random.seed(self.seed)

        self.tracks = list(self.get_tracks())
        if not len(self.tracks):
            raise RuntimeError("No tracks found")

    def __getitem__(self, index):
        # first, get target track
        track_path = self.tracks[index // self.samples_per_track]["path"]
        min_duration = self.tracks[index // self.samples_per_track]["min_duration"]

        if self.split == "train" and self.seq_duration:
            if self.random_chunks:
                # determine start seek by target duration
                start = random.uniform(0, min_duration - self.seq_duration)
            else:
                start = 0

            # assemble the mixture of target and interferers
            audio_sources = []
            # load target
            target_audio, _ = load_audio(
                track_path / self.target_file, start=start, dur=self.seq_duration
            )
            target_audio = self.resampler(target_audio)
            target_audio = self.source_augmentations(target_audio)
            audio_sources.append(target_audio)
            # load interferers
            for source in self.interferer_files:
                # optionally select a random track for each source
                if self.random_track_mix:
                    random_idx = random.choice(range(len(self.tracks)))
                    track_path = self.tracks[random_idx]["path"]
                    if self.random_chunks:
                        min_duration = self.tracks[random_idx]["min_duration"]
                        start = random.uniform(0, min_duration - self.seq_duration)

                audio, _ = load_audio(track_path / source, start=start, dur=self.seq_duration)
                audio = self.resampler(audio)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            stems = torch.stack(audio_sources)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # target is always the first element in the list
            y = stems[0]
        else:
            # load all sources
            x, _ = load_audio(track_path / self.mixture_file)
            x = self.resampler(x)
            y, _ = load_audio(track_path / self.target_file)
            y = self.resampler(y)

        return x, y

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]
                if not all(sp.exists() for sp in source_paths):
                    print("Exclude track ", track_path)
                    continue

                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i["duration"] for i in infos)
                    if min_duration > self.seq_duration:
                        yield ({"path": track_path, "min_duration": min_duration})
                else:
                    yield ({"path": track_path, "min_duration": None})


class MUSDBDataset(UnmixDataset):
    def __init__(
        self,
        target: str = "vocals",
        root: str = None,
        download: bool = False,
        is_wav: bool = False,
        subsets: str = "train",
        split: str = "train",
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        random_track_mix: bool = False,
        # seed: int = 42,
        sample_rate: float = 44100.0,
        *args,
        **kwargs,
    ) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        # import musdb
        
        # self.seed = seed
        # random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=root,
            is_wav=True,
            split=split,
            subsets=subsets,
            download=download,
            *args,
            **kwargs,
        )
        self.sample_rate = sample_rate
        self.resampler = torchaudio.transforms.Resample(44100, 24000)

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        # print(index // self.samples_per_track, index)
        track = self.mus.tracks[index // self.samples_per_track]
        # track = self.mus.tracks[-2]

        # at training time we assemble a custom mix
        if self.split == "train" and self.seq_duration:
            for k, source in enumerate(self.mus.setup["sources"]):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration

                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
                # load source audio and apply time domain source_augmentations
                audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
                if self.sample_rate != 44100:
                    audio = self.resampler(audio)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)
            
            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup["sources"].keys()).index("vocals")
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)
            if self.sample_rate != 44100:
                x = torchaudio.functional.resample(x, 44100, self.sample_rate)
                y = torchaudio.functional.resample(y, 44100, self.sample_rate)
        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )

    parser.add_argument("--root", type=str, help="root path of dataset")

    parser.add_argument(
        "--save", action="store_true", help=("write out a fixed dataset of samples")
    )

    parser.add_argument("--target", type=str, default="vocals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audio-backend",
        type=str,
        default="soundfile",
        help="Set torchaudio backend (`sox_io` or `soundfile`",
    )

    # I/O Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=5.0,
        help="Duration of <=0.0 will result in the full audio",
    )

    parser.add_argument("--batch-size", type=int, default=16)

    args, _ = parser.parse_known_args()

    torchaudio.set_audio_backend(args.audio_backend)

    train_dataset, valid_dataset, args = load_datasets(parser, args)
    print("Audio Backend: ", torchaudio.get_audio_backend())

    # Iterate over training dataset and compute statistics
    total_training_duration = 0
    for k in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[k]
        total_training_duration += x.shape[1] / train_dataset.sample_rate
        if args.save:
            torchaudio.save("test/" + str(k) + "x.wav", x.T, train_dataset.sample_rate)
            torchaudio.save("test/" + str(k) + "y.wav", y.T, train_dataset.sample_rate)

    print("Total training duration (h): ", total_training_duration / 3600)
    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur

    train_sampler = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    for x, y in tqdm.tqdm(train_sampler):
        print(x.shape)
