import os
# Signal processing setup
SR = 16000
FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96

# Machine learning setup
BATCH_SIZE = 1 # (size of the batch during prediction)

# Output labels
MTT_LABELS = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']

MSD_LABELS = ['rock','pop','alternative','indie','electronic','female vocalists','dance','00s','alternative rock','jazz','beautiful','metal','chillout','male vocalists','classic rock','soul','indie rock','Mellow','electronica','80s','folk','90s','chill','instrumental','punk','oldies','blues','hard rock','ambient','acoustic','experimental','female vocalist','guitar','Hip-Hop','70s','party','country','easy listening','sexy','catchy','funk','electro','heavy metal','Progressive rock','60s','rnb','indie pop','sad','House','happy']

MSD_MUSICNN_BIG_LINKS = {
    ".data-00000-of-00001": "https://github.com/jordipons/musicnn/raw/master/musicnn/MSD_musicnn_big/.data-00000-of-00001",
    ".index": "https://github.com/jordipons/musicnn/raw/master/musicnn/MSD_musicnn_big/.index",
    ".meta": "https://github.com/jordipons/musicnn/raw/master/musicnn/MSD_musicnn_big/.meta",
    "config.json": "https://raw.githubusercontent.com/jordipons/musicnn/master/musicnn/MSD_musicnn_big/config.json",
    "checkpoint": "https://raw.githubusercontent.com/jordipons/musicnn/master/musicnn/MSD_musicnn_big/checkpoint",
}

PRETRAIN_FOLDER = f"{os.path.dirname(__file__)}/MSD_musicnn_big"
