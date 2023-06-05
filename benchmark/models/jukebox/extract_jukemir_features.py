"""code adapted from
https://github.com/p-lambda/jukemir
https://github.com/ldzhangyx/simplified-jukemir
https://github.com/openai/jukebox/
"""

import os
import argparse

import wget
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import benchmark.models.jukebox.configuration as config
from benchmark.models.jukebox.make_models import MODELS, make_prior, make_vqvae
from benchmark.models.jukebox.hparams import Hyperparams, setup_hparams
from benchmark.utils.audio_utils import load_audio, find_audios


def round_to_factor_of_sample_length(audio, sample_length):
    # assert single channel
    assert len(audio.shape) == 2 and audio.shape[0] == 1

    sample_length_factor_candidate = [sample_length, sample_length // 2, sample_length // 4, sample_length // 8]

    audio_length = audio.shape[1]

    # select the closest factor
    selected_sample_length = min(sample_length_factor_candidate, key=lambda x: abs(x - audio_length))

    if audio_length < selected_sample_length:
        # pad with torch.zeros
        audio = torch.cat(
            (audio, torch.zeros((1, selected_sample_length - audio_length), dtype=audio.dtype)),
            dim=1,
        )
    elif audio_length > selected_sample_length:
        audio = audio[:, :selected_sample_length]

    return audio


def download_model():
    pretrain_folder = config.PRETRAIN_FOLDER
    if not os.path.exists(pretrain_folder):
        os.makedirs(pretrain_folder)
    
    download_links = config.DOWNLOAD_LINKS

    for model_name, link in download_links.items():
        if not os.path.exists(os.path.join(pretrain_folder, model_name)):
            print(f"Downloading {model_name} from {link}...")
            wget.download(link, out=pretrain_folder)


class JukeboxFeature(nn.Module):
    def __init__(
            self, device='cuda', force_half=False
        ):
        super().__init__()
        model = "5b"  # might not fit to other settings, e.g., "1b_lyrics" or "5b_lyrics"
        self.hps = Hyperparams()
        self.hps.sr = config.JUKEBOX_SAMPLE_RATE
        self.hps.n_samples = 8
        self.hps.name = "samples"
        self.hps.levels = 3
        self.hps.hop_fraction = [0.5, 0.5, 0.125]
        self.vqvae, *priors = MODELS[model]
        hps_1 = setup_hparams(self.vqvae, dict(sample_length=config.SAMPLE_LENGTH))
        hps_1.restore_vqvae = f"{config.PRETRAIN_FOLDER}/vqvae.pth.tar"
        self.vqvae = make_vqvae(
            hps_1, device
        )
        self.force_half = force_half

        # Set up language model
        hps_2 = setup_hparams(priors[-1], dict())
        hps_2["prior_depth"] = config.DEPTH
        hps_2.restore_prior = f"{config.PRETRAIN_FOLDER}/prior_level_2.pth.tar"
        self.top_prior = make_prior(hps_2, self.vqvae, device)
    
    def get_z(self, audio, vqvae, device='cuda'):
        audio = audio[: ].unsqueeze(2)
        zs = vqvae.encode(audio.to(device))

        z = zs[-1].flatten()[np.newaxis, :]

        return z

    def get_cond(self, hps, top_prior, T, device='cuda'):
        sample_length_in_seconds = 62

        hps.sample_length = (
            int(sample_length_in_seconds * hps.sr) // top_prior.raw_to_tokens
        ) * top_prior.raw_to_tokens

        # NOTE: the 'lyrics' parameter is required, which is why it is included,
        # but it doesn't actually change anything about the `x_cond`, `y_cond`,
        # nor the `prime` variables
        metas = [
                    dict(
                        artist="unknown",
                        genre="unknown",
                        total_length=hps.sample_length,
                        offset=0,
                        lyrics="""lyrics go here!!!""",
                    ),
                ] * hps.n_samples

        labels = [None, None, top_prior.labeller.get_batch_labels(metas, device)]

        x_cond, y_cond, prime = top_prior.get_cond(None, top_prior.get_y(labels[-1], 0))

        x_cond = x_cond[0, :T][np.newaxis, ...]
        y_cond = y_cond[0][np.newaxis, ...]

        return x_cond, y_cond

    def get_final_activations(self, z, x_cond, y_cond, top_prior):
        x = z[:, :]

        # make sure that we get the activations
        top_prior.prior.only_encode = True

        # encoder_kv and fp16 are set to the defaults, but explicitly so
        out = top_prior.prior.forward(
            x, x_cond=x_cond, y_cond=y_cond, encoder_kv=None, fp16=self.force_half
        )

        return out

    @torch.no_grad()
    def get_acts_from_wav(self, audio):
        # the time dimension will be downsampled x128
        cur_T = audio.shape[1] // config.DOWNSAMPLE_RATE
        assert config.MAX_T % cur_T == 0, f"After downsampling, the time dim must be divisible by {config.MAX_T}. "

        # run vq-vae on the audio
        z = self.get_z(audio, self.vqvae)

        # get conditioning info
        x_cond, y_cond = self.get_cond(self.hps, self.top_prior, cur_T)

        # get the activations from the LM
        acts = self.get_final_activations(z, x_cond, y_cond, self.top_prior)

        # postprocessing
        acts = acts.squeeze().type(torch.float32)  # [T, H], T = factor of 8192, H = 4800

        return acts

    def forward(self, wav, out_T=config.OUT_T):
        out = self.get_acts_from_wav(wav) # [T, H], T = factor of 8192, H = 4800
        assert config.MAX_T % out_T == 0, f"The 'out_T' param must be divisible by {config.MAX_T}. "
        acts = out.view(out_T, -1, 4800)
        return acts.mean(dim=1, keepdim=True) # [1, 1, H] = [1, 1, 4800]


def select_args(config):
    args = argparse.Namespace()
    args.accelerator = config.dataset.pre_extract.accelerator
    args.output_dir = config.dataset.pre_extract.output_dir
    args.overwrite = config.dataset.pre_extract.overwrite
    args.audio_dir = config.dataset.pre_extract.audio_dir
    args.n_shard = config.args.n_shard
    args.shard_rank = config.args.shard_rank
    args.keep_folder_structure = config.dataset.pre_extract.keep_folder_structure
    args.force_half = config.dataset.pre_extract.feature_extractor.force_half
    args.crop_to_length_in_sec = config.dataset.pre_extract.audio_loader.crop_to_length_in_sec
    return args


def main(config):
    args = select_args(config)

    download_model()

    assert args.accelerator == 'gpu', 'Only GPU is supported for Jukebox'
    device = torch.device('cuda')

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')

    if args.n_shard > 1:
        print(f'processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort() # make sure no intersetction
        audio_files = audio_files[args.shard_rank * len(audio_files) // args.n_shard : (args.shard_rank + 1) * len(audio_files) // args.n_shard]    

    feature_extractor = JukeboxFeature(force_half=args.force_half)
    feature_extractor.eval()
    feature_extractor.to(device)

    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            try:
                waveform = load_audio(
                    audio_file,
                    target_sr= config.JUKEBOX_SAMPLE_RATE,
                    is_mono=True,
                    is_normalize=True, # the same as jukemir
                    # NOTE: jukemir actually crop all audio to 25s for EMO, MTT, GTZAN, GS
                    # you can also crop the audio if the dataset is too long
                    crop_randomly=False,
                    crop_to_length_in_sec=args.crop_to_length_in_sec, 
                    device=device,
                )
            except Exception as e:
                print(f"skip audio {audio_file} because of {e}")
                continue
            
            # NOTE: according to Jukemir, they use a 30s window for feature extraction.
            # but only ~23.77s is used for the actual feature, the rest is discarded.
            # this is because they round the audio sample length to 8192*128=1048576 
            # for efficiency. (1048576 / 44100 = 23.77)
            # 
            # the original audio length is hardcoded, but here we use sliding window 
            # to support long audio, e.g. MTG. 
            # for simplicity and storage efficiency, we do global average pooling.
            # we also use dynamic audio length rounding to support short audio, e.g. 
            # VocalSet.
            wav = waveform
            sliding_window_size_in_sec = 30.0
            overlap_in_sec = 0.0
            wavs = []
            for i in range(0, wav.shape[-1], int(config.JUKEBOX_SAMPLE_RATE * (sliding_window_size_in_sec - overlap_in_sec))):
                wavs.append(wav[:, i : i + int(config.JUKEBOX_SAMPLE_RATE * sliding_window_size_in_sec)])
            # discard the last chunk if it is shorter than 1s
            if wavs[-1].shape[-1] < config.JUKEBOX_SAMPLE_RATE * 1:
                wavs = wavs[:-1]
            features = []
            for wav in wavs:
                # rounds the sample length to be a factor of 1048576
                input_wav = round_to_factor_of_sample_length(wav, config.SAMPLE_LENGTH)
                feature = feature_extractor(input_wav)
                features.append(feature)
            features = torch.cat(features, dim=1).mean(dim=1, keepdim=True)

            # save to npy
            features = features.cpu().numpy()
            if args.keep_folder_structure:
                output_file = os.path.join(
                    args.output_dir,
                    os.path.relpath(audio_file, args.audio_dir)+'.npy',
                )
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            else:
                output_file = os.path.join(
                    args.output_dir,
                    os.path.basename(audio_file)+'.npy',
                )
            if not args.overwrite:
                assert not os.path.exists(output_file), f"{output_file} exists. If you want to overwrite, please add --overwrite."
            np.save(output_file, features)

