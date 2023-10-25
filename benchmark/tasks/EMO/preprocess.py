"""adapted from https://github.com/p-lambda/jukemir/blob/29b701b7891306e7434634f71557cd682ee18505/jukemir/datasets/__init__.py
"""
import csv
import gzip
import json
import shlex
import pathlib
import hashlib
import subprocess
from argparse import ArgumentParser

import numpy as np
from scipy.io.wavfile import read as wavread
from tqdm import tqdm


def compute_checksum(path_or_bytes, algorithm="sha256", gunzip=False, chunk_size=4096):
    """Computes checksum of target path.

    Parameters
    ----------
    path_or_bytes : :class:`pathlib.Path` or bytes
       Location or bytes of file to compute checksum for.
    algorithm : str, optional
       Hash algorithm (from :func:`hashlib.algorithms_available`); default ``sha256``.
    gunzip : bool, optional
       If true, decompress before computing checksum.
    chunk_size : int, optional
       Chunk size for iterating through file.

    Raises
    ------
    :class:`FileNotFoundError`
       Unknown path.
    :class:`IsADirectoryError`
       Path is a directory.
    :class:`ValueError`
       Unknown algorithm.

    Returns
    -------
    str
       Hex representation of checksum.
    """
    if algorithm not in hashlib.algorithms_guaranteed or algorithm.startswith("shake"):
        raise ValueError("Unknown algorithm")
    computed = hashlib.new(algorithm)
    if isinstance(path_or_bytes, bytes):
        computed.update(path_or_bytes)
    else:
        open_fn = gzip.open if gunzip else open
        with open_fn(path_or_bytes, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                computed.update(data)
    return computed.hexdigest()


def write_dataset_json(dataset, path):
    path = pathlib.Path(path)
    if path.suffix == ".gz":
        open_fn = lambda: gzip.open(path, "wt", encoding="utf-8")
    else:
        open_fn = lambda: open(path, "w")
    with open_fn() as f:
        json.dump(dataset, f, indent=2, sort_keys=True)


def run_cmd_sync(cmd, cwd=None, interactive=False, timeout=None):
    """Runs a console command synchronously and returns the results.

    Parameters
    ----------
    cmd : str
       The command to execute.
    cwd : :class:`pathlib.Path`, optional
       The working directory in which to execute the command.
    interactive : bool, optional
       If set, run command interactively and pipe all output to console.
    timeout : float, optional
       If specified, kills process and throws error after this many seconds.

    Returns
    -------
    int
       Process exit status code.
    str, optional
       Standard output (if not in interactive mode).
    str, optional
       Standard error (if not in interactive mode).

    Raises
    ------
    :class:`ValueError`
       Empty command.
    :class:`NotADirectoryError`
       Specified working directory is not a directory.
    :class:`subprocess.TimeoutExpired`
       Specified timeout expired.
    """
    if cmd is None or len(cmd.strip()) == 0:
        raise ValueError()

    kwargs = {}
    if not interactive:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    p = subprocess.Popen(shlex.split(cmd), cwd=cwd, **kwargs)
    try:
        p_res = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        p.kill()
        p.wait()
        raise e

    result = p.returncode

    if not interactive:
        stdout, stderr = [s.decode("utf-8").strip() for s in p_res]
        result = (result, stdout, stderr)

    return result


def iter_emomusic(metadata_only=False):
    d = PATH
    def parse_minsec(s):
        s = s.split(".")
        t = float(s[0]) * 60
        if len(s) > 1:
            assert len(s) == 2
            if len(s[1]) == 1:
                s[1] += "0"
            t += float(s[1])
        return t

    # prase annotations CSV
    audio_uids = set()
    uid_to_metadata = {}
    for stem in [
        "songs_info",
        "static_annotations",
        "valence_cont_average",
        "valence_cont_std",
        "arousal_cont_average",
        "arousal_cont_std",
    ]:
        with open(pathlib.Path(d, f"{stem}.csv"), "r") as f:
            for row in csv.DictReader(f):
                row = {k: v.strip() for k, v in row.items()}
                uid = str(int(row["song_id"])).zfill(4)
                if stem == "songs_info":
                    assert uid not in uid_to_metadata
                    audio_uid = (row["Artist"], row["Song title"])
                    # NOTE: Only one clip per song in this dataset
                    assert audio_uid not in audio_uids
                    audio_uids.add(audio_uid)
                    clip_start = parse_minsec(row["start of the segment (min.sec)"])
                    clip_end = parse_minsec(row["end of the segment (min.sec)"])
                    clip_dur = clip_end - clip_start
                    assert clip_dur == 45.0
                    uid_to_metadata[uid] = {
                        "split": "test"
                        if row["Mediaeval 2013 set"] == "evaluation"
                        else "train",
                        "clip": {
                            "audio_uid": audio_uid,
                            "audio_duration": clip_end,
                            "clip_idx": 0,
                            "clip_offset": clip_start,
                        },
                        "y": None,
                        "extra": {},
                    }
                else:
                    assert uid in uid_to_metadata
                uid_to_metadata[uid]["extra"][stem] = row
                if stem == "static_annotations":
                    uid_to_metadata[uid]["y"] = [
                        float(row["mean_arousal"]),
                        float(row["mean_valence"]),
                    ]

    # Normalize
    arousals = [
        metadata["y"][0]
        for metadata in uid_to_metadata.values()
        if metadata["split"] == "train"
    ]
    valences = [
        metadata["y"][1]
        for metadata in uid_to_metadata.values()
        if metadata["split"] == "train"
    ]
    arousal_mean = np.mean(arousals)
    arousal_std = np.std(arousals)
    valence_mean = np.mean(valences)
    valence_std = np.std(valences)
    for metadata in uid_to_metadata.values():
        metadata["y"] = [
            (metadata["y"][0] - arousal_mean) / arousal_std,
            (metadata["y"][1] - valence_mean) / valence_std,
        ]

    # split train/valid/test
    ratios = ["train"] * 8 + ["valid"] * 2
    for uid, metadata in uid_to_metadata.items():
        if metadata["split"] == "train":
            artist = metadata["extra"]["songs_info"]["Artist"]
            artist = "".join(
                [
                    c
                    for c in artist.lower()
                    if (ord(c) < 128 and (c.isalpha() or c.isspace()))
                ]
            )
            artist = " ".join(artist.split())
            artist_id = int(
                compute_checksum(artist.encode("utf-8"), algorithm="sha1"), 16
            )
            split = ratios[artist_id % len(ratios)]
            metadata["split"] = split

    # Yield unique id, metadata, and path (if downloaded) for each audio clip.
    for uid, metadata in uid_to_metadata.items():
        # Yield result
        result = (uid, metadata)
        if not metadata_only:
            mp3_path = pathlib.Path(d, "clips_45seconds", f"{int(uid)}.mp3")
            result = result + (mp3_path,)
        yield result



if __name__ == "__main__":
    parser = ArgumentParser(description='A simple program that greets the user.')
    parser.add_argument('--dataset_dir', type=str, default="data/EMO",  help='The path of the dataset root')
    args = parser.parse_args()

    PATH = args.dataset_dir
    out_dir = pathlib.Path(f"{PATH}/emomusic")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = pathlib.Path(out_dir, "meta.json")
    audio_dir = pathlib.Path(out_dir, "wav")

    all_metadata = {}
    _tqdm = tqdm  #if progress_bar else lambda x: x
    for example in _tqdm(iter_emomusic()):
        if len(example) == 2:
            uid, metadata = example
        elif len(example) >= 3:
            if len(example) == 3:
                uid, metadata, src_audio_path = example
                clip_offset = None
            elif len(example) == 5:
                uid, metadata, src_audio_path, clip_offset, clip_duration = example
            else:
                raise ValueError("Bad iterator")
            if src_audio_path is not None:
                dest_audio_path = pathlib.Path(audio_dir, f"{uid}.wav")
                dest_audio_path.parent.mkdir(exist_ok=True)
                clip_args = (
                    ""
                    if clip_offset is None
                    else f"-ss {clip_offset} -t {clip_duration}"
                )
                status, stdout, stderr = run_cmd_sync(
                    f"ffmpeg -y -i {src_audio_path} {clip_args} -ac 1 -bitexact {dest_audio_path}",
                    timeout=60,
                )
                try:
                    sr, audio = wavread(dest_audio_path)
                    assert audio.ndim == 1
                    assert audio.shape[0] > 0
                    if "clip" in metadata:
                        metadata["clip"]["clip_duration"] = audio.shape[0] / sr
                except:
                    raise Exception(f"Could not convert source audio to wav:\n{stderr}")
        else:
            raise ValueError("Bad iterator")

        if uid in all_metadata:
            raise ValueError("Duplicate UID in interator")
        all_metadata[uid] = metadata

    write_dataset_json(all_metadata, metadata_path)
