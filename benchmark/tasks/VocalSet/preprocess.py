# Paper: http://ismir2018.ircam.fr/doc/pdfs/114_Paper.pdf
import os
import zipfile
import wget
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import argparse
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42617)

# Choosing '--keep' would aquire highest Acc in Singer Detection Task (Layer 2: 0.8240454),
# otherwise would aquire highest Acc in Technique Detection Task (Layer 8 or 11: 0.7219298).
parser = argparse.ArgumentParser(description='Keep rest of slice less than 3 seconds(default: False)')
parser.add_argument('--keep', help='Keep rest of slice less than 3 seconds',
                    action='store_true')
parser.add_argument('--remove', help='Remove VocalSet.zip',
                    action='store_true')
args = parser.parse_args()


def open_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return sig, sr


# Remove silence
def remove_silence(aud):
    sig, sr = aud[0], aud[1]
    return torchaudio.functional.vad(sig, sr), sr


# Set sample rate at 44.1k
def resample(aud, newsr=44100):
    sig, sr = aud[0], aud[1]
    if (sr == newsr):
        return sig, sr
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    return resig, newsr


# Normalize chunks by means and standard deviation
def normalization(sig):
    return (sig - sig.mean())/sig.std()


# Chunk audio at 3 seconds, rest of slices more than 1 seconds would also be remained
def partition(aud, max_ms=3000):
    sig, sr = aud[0], aud[1]
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms
    chunks = []
    i = -1

    while (sig_len > max_len):
        i += 1
        chunks.append(normalization(sig[:,max_len*i:max_len*(i+1)]))
        sig_len -= max_len
    if (args.keep and sig_len > max_len/3):
      chunks.append(normalization(sig[:,max_len*(i+1):]))

    return chunks


if __name__ == "__main__":
  unzip_dir = 'data/VocalSet/audio/'
  zip_dir = 'data/VocalSet/VocalSet.zip'
  archive = zipfile.ZipFile(zip_dir)


  tech_types = ['belt', 'breathy', 'inhaled', 'lip_trill', 'spoken', 'straight', 'trill', 'trillo', 'vibrato', 'vocal_fry', 'fast_forte', 'fast_piano', 'forte', 'messa', 'pp', 'slow_forte', 'slow_piano']
  # Technique classes that would be utilized in the task
  # Refer: train_singers_technique.txt   
  tech_task_type = ['belt', 'breathy', 'inhaled', 'lip_trill', 'spoken', 'straight', 'trill', 'trillo', 'vibrato', 'vocal_fry']
  filter = ['', '.DS_Store']


  # Unzip files
  print('\nUnzipping files ...')
  for file in tqdm(archive.infolist()):
    file_str = file.filename.split('/')
    if len(file_str) == 5 and (file_str[-2] in tech_types) and (file_str[-1] not in filter):
      file.filename = os.path.basename(file.filename)
      unzip_path = unzip_dir+file_str[-2]
      archive.extract(file, unzip_path)


  # Align the structure of files with 'Table 2' in the paper
  # Complement missing information in file names
  name_correction = [('/lip_trill/lip_trill_arps.wav', '/lip_trill/f2_lip_trill_arps.wav'),
                    ('/lip_trill/scales_lip_trill.wav', '/lip_trill/m3_scales_lip_trill.wav'),
                    ('/straight/arpeggios_straight_a.wav', '/straight/f4_arpeggios_straight_a.wav'),
                    ('/straight/arpeggios_straight_e.wav', '/straight/f4_arpeggios_straight_e.wav'),
                    ('/straight/arpeggios_straight_i.wav', '/straight/f4_arpeggios_straight_i.wav'),
                    ('/straight/arpeggios_straight_o.wav', '/straight/f4_arpeggios_straight_o.wav'),
                    ('/straight/arpeggios_straight_u.wav', '/straight/f4_arpeggios_straight_u.wav'),
                    ('/straight/row_straight.wav', '/straight/m8_row_straight.wav'),
                    ('/straight/scales_straight_a.wav', '/straight/f4_scales_straight_a.wav'),
                    ('/straight/scales_straight_e.wav', '/straight/f4_scales_straight_e.wav'),
                    ('/straight/scales_straight_i.wav', '/straight/f4_scales_straight_i.wav'),
                    ('/straight/scales_straight_o.wav', '/straight/f4_scales_straight_o.wav'),
                    ('/straight/scales_straight_u.wav', '/straight/f4_scales_straight_u.wav'),
                    ('/vocal_fry/scales_vocal_fry.wav', '/vocal_fry/f2_scales_vocal_fry.wav'),
                    ('/fast_forte/arps_fast_piano_c.wav', '/fast_forte/f9_arps_fast_piano_c.wav'),
                    ('/fast_piano/fast_piano_arps_f.wav', '/fast_piano/f2_fast_piano_arps_f.wav'),
                    ('/fast_piano/arps_c_fast_piano.wav', '/fast_piano/m3_arps_c_fast_piano.wav'),
                    ('/fast_piano/scales_fast_piano_f.wav', '/fast_piano/f3_scales_fast_piano_f.wav'),
                    ('/fast_piano/scales_c_fast_piano_a.wav', '/fast_piano/m10_scales_c_fast_piano_a.wav'),
                    ('/fast_piano/scales_c_fast_piano_e.wav', '/fast_piano/m10_scales_c_fast_piano_e.wav'),
                    ('/fast_piano/scales_c_fast_piano_i.wav', '/fast_piano/m10_scales_c_fast_piano_i.wav'),
                    ('/fast_piano/scales_c_fast_piano_o.wav', '/fast_piano/m10_scales_c_fast_piano_o.wav'),
                    ('/fast_piano/scales_c_fast_piano_u.wav', '/fast_piano/m10_scales_c_fast_piano_u.wav'),
                    ('/fast_piano/scales_f_fast_piano_a.wav', '/fast_piano/m10_scales_f_fast_piano_a.wav'),
                    ('/fast_piano/scales_f_fast_piano_e.wav', '/fast_piano/m10_scales_f_fast_piano_e.wav'),
                    ('/fast_piano/scales_f_fast_piano_i.wav', '/fast_piano/m10_scales_f_fast_piano_i.wav'),
                    ('/fast_piano/scales_f_fast_piano_o.wav', '/fast_piano/m10_scales_f_fast_piano_o.wav'),
                    ('/fast_piano/scales_f_fast_piano_u.wav', '/fast_piano/m10_scales_f_fast_piano_u.wav'),]

  # Delete duplicate files and files not mentioned in 'Table 2'.                   
  file_delete = ['/vibrato/f2_scales_vibrato_a(1).wav',
                '/vibrato/caro_vibrato.wav',
                '/vibrato/dona_vibrato.wav',
                '/vibrato/row_vibrato.wav',
                '/vibrato/slow_vibrato_arps.wav']

  for i in range(len(name_correction)):
    if os.path.exists(unzip_dir+name_correction[i][0]):
      os.rename(unzip_dir+name_correction[i][0], unzip_dir+name_correction[i][1])
  for i in range(len(file_delete)):
    if os.path.exists(unzip_dir+file_delete[i]):
      os.remove(unzip_dir+file_delete[i])


  data_df = pd.DataFrame({'file_name':[], 'tech_class':[], 'singer_class':[], 'file_id':[]})

  # Scan all wave files and process according to the preprocessing methods in original paper
  # Process: remove silence, set sample rate, chunk into 3 seconds slices, normalize chunks
  # Name of the file, technique class, singer name, file id (to divide dataset) are recorded in data_df (audio_data.csv)
  print('\nProcessing data ...')
  for cur_type in tqdm(sorted(os.listdir(unzip_dir))):
    files = sorted(os.listdir(unzip_dir+cur_type))
    for i, file in enumerate(files):
      chunks = partition(resample(remove_silence(open_file(unzip_dir+cur_type+'/'+file))))
      for j in range(len(chunks)):
        new_file = cur_type+'/'+file.split('.')[0]+'_'+(str(j).zfill(2))+'.wav'
        torchaudio.save(unzip_dir+new_file, chunks[j], 44100)
        data_df.loc[len(data_df)] = [new_file, cur_type, file.split('_')[0], int(i)]
      os.remove(unzip_dir+cur_type+'/'+file)

  # --------For technique identification--------
  # Divide technique dataset into training set and the test set according to the singer
  tech_data_df = data_df[data_df['tech_class'].isin(tech_task_type)].reset_index(drop=True)
  print('\n\nSplitting datasets and generating .txt files')
  train_type = ['f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f9', 'm1', 'm2', 'm4', 'm6', 'm7', 'm8', 'm9', 'm11']
  test_type  = ['f2', 'f8', 'm3', 'm5', 'm10']
  train_val_df = tech_data_df[tech_data_df['singer_class'].isin(train_type)].reset_index(drop=True)
  test_df  = tech_data_df[tech_data_df['singer_class'].isin(test_type)].reset_index(drop=True)    

  # Randomly devide training set into train & valid sets (recordings are disjoint, meaning that parts of the same recording were not put in both)
  # Fix random seed to make the shuffle repeatable
  gss = GroupShuffleSplit(n_splits=1, test_size=0.4)
  train_id, valid_id = next(gss.split(train_val_df, groups=train_val_df.file_id))
  train_df, valid_df = train_val_df.iloc[train_id].reset_index(drop=True), train_val_df.iloc[valid_id].reset_index(drop=True)

  np.savetxt('data/VocalSet/train_t.txt', train_df.file_name, fmt='%s')
  np.savetxt('data/VocalSet/valid_t.txt', valid_df.file_name, fmt='%s')
  np.savetxt('data/VocalSet/test_t.txt', test_df.file_name, fmt='%s')


  # --------For singer identification--------
  # Randomly devide data set into train, valid and test sets (recordings are disjoint, meaning that parts of the same recording were not put in both)
  # Fix random seed to make the shuffle repeatable
  gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
  train_val_id, test_id = next(gss.split(data_df, groups=data_df.file_id))
  train_val_df, test_df = data_df.iloc[train_val_id].reset_index(drop=True), data_df.iloc[test_id].reset_index(drop=True)
  gss = GroupShuffleSplit(n_splits=1, test_size=0.4)
  train_id, valid_id = next(gss.split(train_val_df, groups=train_val_df.file_id))
  train_df, valid_df = train_val_df.iloc[train_id].reset_index(drop=True), train_val_df.iloc[valid_id].reset_index(drop=True)

  np.savetxt('data/VocalSet/train_s.txt', train_df.file_name, fmt='%s')
  np.savetxt('data/VocalSet/valid_s.txt', valid_df.file_name, fmt='%s')
  np.savetxt('data/VocalSet/test_s.txt', test_df.file_name, fmt='%s')


  if args.remove:
    os.remove(down_file_path)
    print('\nVocalSet.zip has been removed.')

  print('\nPreprocess Done')