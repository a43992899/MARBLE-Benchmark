"""minimal code to load mule model and extract features
"""
import os

import librosa
import tensorflow as tf
import torchaudio
import torch
from scipy.special import logsumexp
import numpy as np

def load_model(model_location="./supporting_data/model/"):
    """
    Loads the model that this feature uses from a configured location.
    """
    # Otherwise assume local
    model = tf.keras.models.load_model(model_location, compile=False)
    return model


def load_audio(audio_path):
    # load audio, convert to mono, resample to 16kHz
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)
    audio = audio.numpy().squeeze()
    return audio


def to_mel(audio):
    # convert to mel spectrogram
    mel_data = librosa.feature.melspectrogram(
            y=audio, 
            sr=16000, 
            n_fft=2048, 
            hop_length=160, 
            win_length=400, 
            window='hann', 
            center=True, 
            pad_mode='reflect', 
            power=2.0,
            n_mels=96,
            fmin=0.0,
            fmax=8000.0,
            norm=2.0,
            htk=True,
            dtype=np.float32
        )
    
    _COMPRESSION_FUNCS = {
        '10log10': lambda x: 10*np.log10(x),
        'log10_nonneg': lambda x: np.log10(10000.0*x + 1.0),
        'log': lambda x: np.log(x),
        'linear': lambda x: x,
        None: lambda x: x,
    }

    mel_data = _COMPRESSION_FUNCS['log10_nonneg'](mel_data)

    _SMALLEST_MAGNITUDE = -9999999999.0

    _mag_range = None

    mel_data = np.nan_to_num(mel_data, nan=_SMALLEST_MAGNITUDE, posinf=_SMALLEST_MAGNITUDE, neginf=_SMALLEST_MAGNITUDE)

    if _mag_range is not None:
            max_val = np.amax(mel_data)
            mel_data = mel_data - max_val
            mel_data = np.maximum(mel_data, - _mag_range)
    """
    array([[4.710301  , 3.867197  , 3.9037328 , ..., 4.585724  , 5.069617  ,
        5.5419006 ],
       [4.479547  , 4.2201114 , 4.250828  , ..., 5.3469596 , 5.279555  ,
        5.4070044 ],
       [3.959253  , 4.176624  , 4.3229537 , ..., 5.839459  , 4.9143186 ,
        5.497674  ],
       ...,
       [1.784141  , 1.4244846 , 1.3697608 , ..., 1.428422  , 1.7329496 ,
        1.9693747 ],
       [0.7733364 , 0.23160188, 0.26645002, ..., 0.45548382, 0.47081986,
        2.0644574 ],
       [0.8182721 , 0.02385915, 0.0075119 , ..., 0.01275367, 0.02037387,
        2.0049872 ]], dtype=float32)
    """
    return mel_data 


def extract_range(feature, start_index=None, end_index=None):
    """
    Extracts data over a given index range from a single feature. This will
    extract regularly spaced 2D slices of data from the input feature. Note
    that first feature will be extracted at an integer multiple of the extractor's
    configured `hop` parameter from the beginning of the feature.

    Args:
        feature: mule.feature.Feature - A feature to extract data from.

        start_index: int - The first index (inclusive) at which to start extracting
        slices.

        end_index: int - The last index (exclusive) at which to return data.

    Return:
        numpy.ndarray - The extracted feature data. Time on the first axis, features
        on the remaining axes.
    """
    _hop = 200
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = feature.shape[1]

    indices = [time for time in range(0, end_index, _hop) if time > start_index]
    # [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
    features = [feature]*len(indices)
    if len(features)==0:
        return np.empty((0,0))
    return extract_batch(features, indices)


def extract_batch(features, indices):
    """
    Extracts a batch of slices of data from a range of features, centered on specific 
    indices.

    Args:
        features: list(mule.features.Feature) - A list of features from which to extract 
        data from.

        indices: list(int) - A list of indices, the same size as `features`. Each element
        provides an index at which to extract data from the coressponding element in the
        `features` argument.

    Return:
        np.ndarray - A batch of features, with features on the batch dimension on the first
        axis and feature data on the remaining axes.
    """
    _look_backward = 150
    _look_forward = 150
    _standard_normalize = True

    indices = [idx if idx >= _look_backward else _look_backward for idx in indices]
    indices = [idx if idx <= feat.shape[-1] - _look_forward else feat.shape[-1] - _look_forward for idx, feat in zip(indices, features)]
    samples = [feature[:, (idx - _look_backward):(idx + _look_forward)] for idx, feature in zip(indices, features)]
    samples = [x.reshape((1, *x.shape, 1)) for x in samples] # Add batch and channel dimensions 
    samples = np.vstack(samples)

    if _standard_normalize:
        samples -= np.mean(samples, axis=(1,2,3), keepdims=True)
        all_vars = np.std(samples, axis=(1,2,3), keepdims=True)
        all_vars = np.nan_to_num(all_vars, nan=1.0, posinf=1.0, neginf=1.0)
        all_vars = np.maximum(all_vars, 0.01)
        samples /= all_vars 

    return samples


def extract_embeddings(model, mel_data):
    _apply_softmax = False

    data = extract_range(mel_data)

    data = model.predict(
            [data],
            callbacks=None,
            verbose=0,
        )
    data = data.T
    if _apply_softmax:
        data = np.nan_to_num(np.exp(data - logsumexp(data, axis=0)))
    return data


def run_one(input_path='./supporting_data/blues.00000.wav', output_path='./supporting_data/blues.00000.wav.npy'):
    model = load_model()
    audio = load_audio(input_path) # [480214]
    mel_data = to_mel(audio) # [96, 3002]
    embeddings = extract_embeddings(model, mel_data) # [1728, 15]
    np.save(output_path, embeddings)


if __name__ == '__main__':
    run_one()