import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
download_dir = os.path.join(base_dir, 'xcodec_mini_infer')
sys.path.append(download_dir)

##########
cwd = os.getcwd()
target = os.path.join(cwd, 'xcodec_mini_infer', 'semantic_ckpts', 'hf_1_325000')
source = os.path.join(base_dir, 'xcodec_mini_infer', 'semantic_ckpts', 'hf_1_325000')

if not os.path.exists(target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    os.symlink(source, target)
    print(f"Symlink created: {target} -> {source}")
##########


from transformers import AutoModelForCausalLM
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from benchmark.utils.audio_utils import load_audio, find_audios
from models.soundstream_hubert_new import SoundStream



def select_args(config):
    args = argparse.Namespace()
    args.accelerator = config.dataset.pre_extract.accelerator
    args.output_dir = config.dataset.pre_extract.output_dir
    args.overwrite = config.dataset.pre_extract.overwrite
    args.audio_dir = config.dataset.pre_extract.audio_dir
    args.n_shard = getattr(config.args, 'n_shard', 1)
    args.shard_rank = getattr(config.args, 'shard_rank', 0)
    args.keep_folder_structure = config.dataset.pre_extract.keep_folder_structure
    # args.stage1_model = '/datapool/data2/home/ruihan/storage/debug/all_m4m/opensuno/MARBLE-Benchmark/benchmark/models/opensuno/stage1.exp31.8.30B.hf_ckpt'
    args.stage1_model = os.path.join(base_dir, 'stage1')
    args.basic_model_config = os.path.join(download_dir, 'final_ckpt', 'config.yaml')
    args.resume_path = os.path.join(download_dir, 'final_ckpt', 'ckpt_00360000.pth')
    args.target_sr = getattr(config.dataset.pre_extract.feature_extractor.pretrain, 'target_sr', 16000)
    args.batch_size = getattr(config.dataset.pre_extract.feature_extractor, 'batch_size', 8)
    # args.tokenizer_path = '/datapool/data2/home/ruihan/storage/debug/all_m4m/opensuno/MARBLE-Benchmark/benchmark/models/opensuno/mm_tokenizer_v0.2_hf/tokenizer.model'
    args.tokenizer_path = os.path.join(base_dir, 'stage1', 'tokenizer.model')
    args.layer = getattr(config.dataset.pre_extract.feature_extractor, 'layer', 30)
    args.sliding_window_size_in_sec = getattr(config.dataset.pre_extract, 'sliding_window_size_in_sec', 30)
    args.sliding_window_overlap_in_percent = getattr(config.dataset.pre_extract, 'sliding_window_overlap_in_percent', 0)
    return args

class YueFeatureExtractor:
    def __init__(self, model_path, tokenizer_path, codec_config, codec_ckpt, device):
        self.device = device
        self.mmtokenizer = _MMSentencePieceTokenizer(tokenizer_path)
        self.model = self._load_model(model_path)
        self.codectool = CodecManipulator("xcodec", 0, 1)
        self.codec_model = self._load_codec_model(codec_config, codec_ckpt)

    def _load_model(self, model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        model.eval()
        return model

    # def _load_codec_model(self, config_path, ckpt_path):
    #     config = OmegaConf.load(config_path)
    #     model = eval(config.generator.name)(**config.generator.config).to(self.device)
    #     model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['codec_model'])
    #     model.eval()
    #     return model

    def _load_codec_model(self, config_path, ckpt_path):
        config = OmegaConf.load(config_path)

        semantic_model_path = config.generator.config.get('semantic_model_path', None)
        if semantic_model_path and semantic_model_path.startswith('./'):
            config.generator.config['semantic_model_path'] = os.path.join(os.path.dirname(config_path), semantic_model_path)

        model = eval(config.generator.name)(**config.generator.config).to(self.device)
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['codec_model'])
        model.eval()
        return model


    def extract_all_features(self, audio_batch):
        with torch.no_grad():
            codes_batch = self.codec_model.encode(audio_batch.to(self.device), target_bw=0.5)
            codes_batch = codes_batch.transpose(0, 1)

            hidden_states = self.model(
                input_ids=torch.as_tensor([self.codectool.npy2ids(c) for c in codes_batch.cpu().numpy()], device=self.device),
                output_hidden_states=True,
                return_dict=True
            ).hidden_states

        return [h.to(dtype=torch.float32) for h in hidden_states]

    def extract_features(self, audio_batch, args):

        sliding_window_size_in_sec = args.sliding_window_size_in_sec
        sliding_window_overlap_in_percent = args.sliding_window_overlap_in_percent

        if sliding_window_size_in_sec:
            assert sliding_window_size_in_sec > 0, "sliding_window_size_in_sec must be positive"
            overlap_in_sec = sliding_window_size_in_sec * sliding_window_overlap_in_percent / 100

            window_size = int(args.target_sr * sliding_window_size_in_sec)
            step_size = int(args.target_sr * (sliding_window_size_in_sec - overlap_in_sec))

            wavs = []
            for i in range(0, audio_batch.shape[-1], step_size):
                # chunk = audio_batch[:, i: i + window_size]
                chunk = audio_batch[:, :, i: i + window_size]
                if chunk.shape[-1] < args.target_sr:
                    break
                wavs.append(chunk)

            features = []
            for wav in wavs:
                features.append(self._extract_single_feature(wav, args.layer))

            features = torch.cat(features, dim=1)
        else:
            features = self._extract_single_feature(audio_batch, args.layer)

        return features

    def _extract_single_feature(self, audio_batch, layer):
        with torch.no_grad():
            codes_batch = self.codec_model.encode(audio_batch.to(self.device), target_bw=0.5)
            codes_batch = codes_batch.transpose(0, 1)
            hidden_states = self.model(
                input_ids=torch.as_tensor([self.codectool.npy2ids(c) for c in codes_batch.cpu().numpy()], device=self.device),
                output_hidden_states=True,
                return_dict=True
            ).hidden_states
        
        return hidden_states[layer].to(dtype=torch.float32)

def main(config):
    args = select_args(config)
    device = torch.device('cuda' if args.accelerator == 'gpu' else args.accelerator)
    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = find_audios(args.audio_dir)
    print(f'Found {len(audio_files)} audio files')

    if args.n_shard > 1:
        print(f'Processing shard {args.shard_rank} of {args.n_shard}')
        audio_files.sort()
        audio_files = audio_files[args.shard_rank * len(audio_files) // args.n_shard : (args.shard_rank + 1) * len(audio_files) // args.n_shard]

    extractor = YueFeatureExtractor(
        model_path=args.stage1_model,
        tokenizer_path=args.tokenizer_path,
        codec_config=args.basic_model_config,
        codec_ckpt=args.resume_path,
        device=device
    )

    target_layers = [args.layer]
    with torch.no_grad():
        for audio_file in tqdm(audio_files):
            try:
                waveform = load_audio(audio_file, target_sr=args.target_sr, is_mono=True, is_normalize=True, device=device)
                waveform = waveform.unsqueeze(0)
            except Exception as e:
                print(f"Skipping {audio_file} due to error: {e}")
                continue

            # all_hidden_states = extractor.extract_features(waveform, args)
            hidden_states = extractor.extract_features(waveform, args)
            # for layer in target_layers:
            layer = args.layer
            # features = all_hidden_states[layer].cpu().numpy()
            features = hidden_states.cpu().numpy()
            features = features.mean(axis=1)

            layer_output_dir = os.path.join(args.output_dir, f"layer_{layer}")
            os.makedirs(layer_output_dir, exist_ok=True)

            if args.keep_folder_structure:
                output_file = os.path.join(layer_output_dir, os.path.relpath(audio_file, args.audio_dir) + '.npy')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            else:
                output_file = os.path.join(layer_output_dir, os.path.basename(audio_file) + '.npy')

            if not args.overwrite:
                assert not os.path.exists(output_file), f"{output_file} exists. Use --overwrite to overwrite."
            np.save(output_file, features)