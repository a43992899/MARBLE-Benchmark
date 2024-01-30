'''
    ALT (Automatic Lyrics Transcription) preprocessing script
    Author: Jiawen Huang
    Comment: Segment the songs into lines
'''

import argparse
import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

mtg_audio_path = "./data/MTG/audio-low/"

if os.path.exists("./data/ALT/") == False:
    os.mkdir("./data/ALT/")
    os.mkdir("./data/ALT/train/")
    os.mkdir("./data/ALT/valid/")
    os.mkdir("./data/ALT/test/")

# the following 3 meta files are the aligned & filtered train/valid/test sets
meta_test = "./benchmark/tasks/ALT/metadata/test.meta"
meta_train = "./benchmark/tasks/ALT/metadata/train.meta"
meta_valid = "./benchmark/tasks/ALT/metadata/valid.meta"

warning_log = "./benchmark/tasks/ALT/metadata/preprocess_log.txt"
write_log = "./benchmark/tasks/ALT/metadata/failed_list.txt"
failed_list = []

def sox_one(args):
    '''
    run segmentation with sox
    '''
    sox_cmd_tmp = "sox {} -G -t wav -r 24000 -c 1 {} trim {} {} 2>>" + warning_log

    save_file, input_file, start, end = args
    if os.path.exists(save_file):
        return True
    # execute
    sox_cmd = sox_cmd_tmp.format(input_file, save_file, start, end-start)
    os.system(sox_cmd)

    return True

def prepare_audio(dataset_dir, meta, save_folder):

    df = pd.read_csv(meta)
    args_list = []
    for idx in df.index:
        id = df.loc[idx, "id"]
        save_file = os.path.join(dataset_dir, save_folder, id+".wav")
        input_file = os.path.join(mtg_audio_path, df.loc[idx, "file"])
        if os.path.exists(input_file) == False:
            failed_list.append(id)
            continue
        start = df.loc[idx, "start"]
        end = df.loc[idx, "end"]
        args_list.append([save_file, input_file, start, end])

    # multiprocessing
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(sox_one, args_list), total=len(args_list)))

def main(args):

    prepare_audio(args.dataset_dir, meta_valid, './valid/')
    prepare_audio(args.dataset_dir, meta_train, './train/')
    prepare_audio(args.dataset_dir, meta_test, './test/')

    with open(write_log, "w") as f:
        for file in failed_list:
            f.write(file+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default="./data/ALT/", help="The path to the dataset directory.")

    args = parser.parse_args()
    main(args)
