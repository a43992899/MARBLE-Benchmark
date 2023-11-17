from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch

import typing as tp


# original encode functionsdef 

def get_audio_seg_embedding(model, frame_x: torch.Tensor):
    length = frame_x.shape[-1]
    duration = length / model.sample_rate
    assert model.segment is None or duration <= 1e-5 + model.segment
    
    if model.normalize:
        mono = frame_x.mean(dim=1, keepdim=True)
        volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
        scale = 1e-8 + volume
        frame_x = frame_x / scale
        scale = scale.view(-1, 1)
    else:
        scale = None
        
    emb = model.encoder(frame_x)
    return emb

def get_audio_encodec_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    
    assert x.dim() == 3
    _, channels, length = x.shape
    assert channels > 0 and channels <= 2
    segment_length = model.segment_length
    # print(model.segment_length) # None，所以返回的就是所有的 dimention
    if segment_length is None:
        segment_length = length
        stride = length
    else:
        stride = model.segment_stride  # type: ignore
        assert stride is not None

    encoded_frame_embedings: tp.List[torch.Tensor] = []
    for offset in range(0, length, stride):
        frame = x[:, :, offset: offset + segment_length]
        print(frame.shape)
        frame_embedding = get_audio_seg_embedding(model, frame)
        encoded_frame_embedings.append(frame_embedding)


    # return mean pooling according to time dimension
    if len(encoded_frame_embedings) == 1:
        # print(encoded_frame_embedings[0].shape) # [1, 128, 2251]
        return torch.mean(encoded_frame_embedings[0], dim=-1) # [1, 128] or # [B, 128]
    else:
        # print(torch.stack(encoded_frame_embedings).shape) # [N, 1, 128, 2251 (n samples)]
        return torch.mean(torch.cat(encoded_frame_embedings, dim=-1), dim=-1) # [B, 128]

    
device = 'cuda'
audio_path = './data/GTZAN/genres/blues/blues.00000.wav'

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device)

model.to(device)
    
# Load and pre-process the audio waveform
wav, sr = torchaudio.load(audio_path)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.unsqueeze(0) # [1, 1, n_samples], 第一个维度是 batch_size
wav = wav.to(device)


# Extract discrete codes from EnCodec
with torch.no_grad():
    wav_encodec_embs = get_audio_encodec_embeddings(model, wav) # model.encode(wav)

print(wav_encodec_embs.shape)
