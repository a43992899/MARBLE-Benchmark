"""adapted from https://github.com/p-lambda/jukemir/blob/29b701b7891306e7434634f71557cd682ee18505/jukemir/datasets/__init__.py
"""
import copy
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


_ASSET_PATHS = set()
_ASSETS = {}
json_path = "benchmark/tasks/GS/assets/giantsteps.json"
with open(json_path, "r") as f:
    d = json.load(f)
for tag, asset in d.items():
    if tag != tag.upper():
        raise AssertionError("Tags should be uppercase")
    if "checksum" not in asset:
        raise AssertionError("Missing checksum")
    try:
        asset["path_rel"] = pathlib.PurePosixPath(asset["path_rel"].strip())
    except:
        raise AssertionError("Invalid path")
    if asset["path_rel"] in _ASSET_PATHS:
        raise AssertionError("Duplicate path")
    _ASSET_PATHS.add(asset["path_rel"])
    asset["path"] = pathlib.Path("data/GS", str(asset["path_rel"])[20:]).resolve()
_ASSETS.update(d)

def _iter_giantsteps(metadata_only=False, clip_duration=None):
    d = PATH
    uids = set()
    for split_name in ["train", "test"]:
        if split_name == "train":
            code = pathlib.Path(
                d,
                f"giantsteps-mtg-key-dataset-fd7b8c584f7bd6d720d170c325a6d42c9bf75a6b",
            )
            mp3_asset_template = "GIANTSTEPS_MTG_KEY_{}"
        else:
            code = pathlib.Path(
                d,
                f"giantsteps-key-dataset-c8cb8aad2cb53f165be51ea099d0dc75c64a844f",
            )
            mp3_asset_template = "GIANTSTEPS_KEY_{}"

        # save metadata & annotation to dictionary, split train/valid
        if split_name == "train":
            did_to_metadata = {}
            with open(
                pathlib.Path(code, "annotations", "beatport_metadata.txt"), "r"
            ) as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    did_to_metadata[int(row["ID"])] = row

            # NOTE: This seemingly-arbitrary split induces target of 80/20
            ratios = (["train"] * 100) + (["valid"] * 16)
            artist_to_split = {}
            for metadata in did_to_metadata.values():
                artists = [a.strip() for a in metadata["ARTIST"].strip().split(",")]
                artist_ids = [
                    int(
                        compute_checksum(
                            a.lower().encode("utf-8"), algorithm="sha1"
                        ),
                        16,
                    )
                    for a in artists
                ]
                artist_splits = [ratios[i % len(ratios)] for i in artist_ids]
                for artist, split in zip(artists, artist_splits):
                    artist_to_split[artist] = split

            # All collaborators of valid artists are valid (run twice for two-hop)
            for _ in range(2):
                for metadata in did_to_metadata.values():
                    artists = [
                        a.strip() for a in metadata["ARTIST"].strip().split(",")
                    ]
                    artist_splits = [artist_to_split[a] for a in artists]
                    if "valid" in artist_splits:
                        for a in artists:
                            artist_to_split[a] = "valid"

            did_to_annotations = {}
            with open(
                pathlib.Path(code, "annotations", "annotations.txt"), "r"
            ) as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    did_to_annotations[int(row["ID"])] = row

        # iterates over all audio-md5 files
        for path in pathlib.Path(code, "md5").glob("*.md5"):
            # extracts info from filename, stored in "uid"
            did = int(path.stem.split(".")[0])
            uid = str(did).zfill(7)
            assert uid not in uids
            uids.add(uid)

            # reads annotations & metadata from text files, stored in "extra"
            extra = {"id": did}
            if split_name == "train":
                extra["beatport_metadata"] = did_to_metadata[did]
                extra["annotations"] = did_to_annotations[did]

            for annotation in [
                "genre",
                "key",
                "jams",
                "giantsteps.genre",
                "giantsteps.key",
            ]:
                if annotation == "jams" and split_name == "train":
                    continue
                if "." in annotation:
                    adir, aext = annotation.split(".")
                else:
                    adir = annotation
                    aext = annotation
                path = pathlib.Path(code, "annotations", adir, f"{did}.LOFI.{aext}")
                with open(path, "r") as f:
                    contents = f.read()
                if annotation == "jams":
                    contents = json.loads(contents)
                extra[annotation] = contents

            # If the dataset is the "train" dataset, the function filters out some audio files based on their annotations and metadata, and determines the split (train/validation) for each audio file based on the artists associated with the file.
            # The function constructs a metadata dictionary containing the split, key signature, and various annotations and metadata associated with the audio file, and yields this metadata to the caller.
            if split_name == "train":
                # NOTE: Skips low-confidence as in (Korzeniowski and Widmer 2018)
                if int(extra["annotations"]["C"]) != 2:
                    continue
                # NOTE: Skips multiple keys as in (Korzeniowski and Widmer 2018)
                if "/" in extra["annotations"]["MANUAL KEY"]:
                    continue
                tonic, scale = extra["annotations"]["MANUAL KEY"].split()
                assert extra["key"].startswith(" ".join((tonic.lower(), scale)))
                enharmonic = {
                    "C#": "Db",
                    "D#": "Eb",
                    "F#": "Gb",
                    "G#": "Ab",
                    "A#": "Bb",
                }
                tonic = enharmonic.get(tonic, tonic)

                artists = [
                    a.strip()
                    for a in extra["beatport_metadata"]["ARTIST"].strip().split(",")
                ]
                artist_splits = [artist_to_split[a] for a in artists]
                assert len(set(artist_splits)) == 1
                induced_split = artist_splits[0]
            else:
                tonic, scale = extra["key"].split()
                induced_split = "test"
            y = " ".join((tonic, scale))

            metadata = {"split": induced_split, "y": y, "extra": extra}

            mp3_asset_tag = mp3_asset_template.format(did)

            if clip_duration is not None:
                mp3_path = _ASSETS[mp3_asset_tag]["path"]
                status, stdout, stderr = run_cmd_sync(
                    f"ffprobe -i {mp3_path} -show_entries format=duration",
                    timeout=60,
                )
                assert status == 0
                duration = float(stdout.strip().splitlines()[1].split("=")[1])
                metadata["clip"] = {
                    "audio_uid": uid,
                    "audio_duration": duration,
                    "clip_idx": None,
                    "clip_offset": None,
                }
                for clip_idx, clip_offset in enumerate(
                    np.arange(0, duration, clip_duration)
                ):
                    this_clip_duration = min(duration - clip_offset, clip_duration)
                    assert (
                        this_clip_duration >= 0
                        and this_clip_duration <= clip_duration
                    )
                    if this_clip_duration < clip_duration:
                        continue
                    clip_uid = f"{uid}-{clip_idx}"
                    clip_metadata = copy.deepcopy(metadata)
                    clip_metadata["clip"]["clip_idx"] = clip_idx
                    clip_metadata["clip"]["clip_offset"] = clip_offset
                    result = (
                        clip_uid,
                        clip_metadata,
                        mp3_path,
                        clip_offset,
                        clip_duration,
                    )
                    yield result[: 2 if metadata_only else 5]


if __name__ == "__main__":
    parser = ArgumentParser(description='A simple program that greets the user.')
    parser.add_argument('--dataset_dir', type=str, default="data/GS",  help='The path of the dataset root')
    args = parser.parse_args()

    PATH = args.dataset_dir
    out_dir = pathlib.Path(f"{PATH}/giantsteps_clips")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = pathlib.Path(out_dir, "meta.json")
    audio_dir = pathlib.Path(out_dir, "wav")

    all_metadata = {}
    _tqdm = tqdm  #if progress_bar else lambda x: x
    for example in _tqdm(_iter_giantsteps(metadata_only=False, clip_duration=30.0)):
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
    