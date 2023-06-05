import os

OUT_T = 1  # how many slices in time to output. 8192 should be divisible by this number.

PRETRAIN_FOLDER = f"{os.path.dirname(__file__)}/downloads"
USING_CACHED_FILE = False

JUKEBOX_SAMPLE_RATE = 44100
SAMPLE_LENGTH = 1048576  # 8192*128=1048576. ~23.77s, same as jukemir
MAX_T = 8192  # 1048576 // 128 
DOWNSAMPLE_RATE = 128
DEPTH = 36

DOWNLOAD_LINKS = {
    "vqvae.pth.tar": "https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar",
    "prior_level_2.pth.tar": "https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar",
}
