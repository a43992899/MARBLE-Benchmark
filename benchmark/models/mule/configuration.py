import os

DOWNLOAD_URLS = {
    "supporting_data/model/saved_model.pb": 'https://github.com/PandoraMedia/music-audio-representations/raw/main/supporting_data/model/saved_model.pb',
    "supporting_data/model/keras_metadata.pb": 'https://github.com/PandoraMedia/music-audio-representations/raw/main/supporting_data/model/keras_metadata.pb',
    "supporting_data/model/variables/variables.data-00000-of-00001": 'https://github.com/PandoraMedia/music-audio-representations/raw/main/supporting_data/model/variables/variables.data-00000-of-00001',
    "supporting_data/model/variables/variables.index": 'https://github.com/PandoraMedia/music-audio-representations/raw/main/supporting_data/model/variables/variables.index',
}

PRETRAIN_FOLDER = os.path.dirname(__file__)

SAMPLE_RATE = 16000
