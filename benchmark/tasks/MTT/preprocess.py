import os
import sys
import glob

import numpy as np
import pandas as pd


class Split:
    def __init__(self):
        print('start preprocessing')

    def get_csv_dataframe(self):
        self.csv_df = pd.read_csv(
            os.path.join(
                self.data_path,
                'MTT',
                'annotations_final.csv'),
            header=None,
            index_col=None,
            sep='\t')
        # sep='\t'

    def get_top50_tags(self):
        # """saving the tags str in a npy file and return the index of top_50 (index_col=None)"""
        tags = list(self.csv_df.loc[0][1:-1])
        tag_count = [np.array(self.csv_df[i][1:], dtype=int).sum()
                     for i in range(1, 189)]
        top_50_tag_index = np.argsort(tag_count)[::-1][:50]
        top_50_tags = np.array(tags)[top_50_tag_index]
        np.save(
            open(
                os.path.join(
                    self.data_path,
                    'MTT',
                    'tags.npy'),
                'wb'),
            top_50_tags)
        return top_50_tag_index

    def write_tags(self, top_50_tag_index):
        binary = np.zeros((25863, 50))
        titles = []
        idx = 0
        for i in range(1, 25864):
            features = np.array(
                self.csv_df.loc[i][top_50_tag_index + 1], dtype=int)
            title = self.csv_df.loc[i][189]  # subdataset/finename.mp3
            binary[idx] = features
            idx += 1
            titles.append(title)

        binary = binary[:len(titles)]
        np.save(
            open(
                os.path.join(
                    self.data_path,
                    'MTT',
                    'binary_label.npy'),
                'wb'),
            binary)
        return titles, binary

    def split(self, titles, binary):
        # """titles: list of file title, binary: numpy labels with shape=(num_data, 50)"""
        tr = []
        val = []
        test = []
        for i, title in enumerate(titles):
            if int(title[0], 16) < 12:  # int("a", 16) = 10
                # if binary[i].sum() > 0:
                tr.append(str(i) + '\t' + title)
            elif int(title[0], 16) < 13:
                # if binary[i].sum() > 0:
                val.append(str(i) + '\t' + title)
            else:
                # if binary[i].sum() > 0:
                test.append(str(i) + '\t' + title)
        self.to_tsv(tr, val, test)

    def to_tsv(self, tr, val, test):
        train_tsv_path = os.path.join(self.data_path, 'MTT', 'train.tsv')
        valid_tsv_path = os.path.join(self.data_path, 'MTT', 'valid.tsv')
        test_tsv_path = os.path.join(self.data_path, 'MTT', 'test.tsv')
        with open(train_tsv_path, 'w') as f:
            for line in tr:
                f.write(line + '\n')
        with open(valid_tsv_path, 'w') as f:
            for line in val:
                f.write(line + '\n')
        with open(test_tsv_path, 'w') as f:
            for line in test:
                f.write(line + '\n')
        print('done')

    def run(self, data_path):
        # f{data_path}/MTT should be the path of MagTagATone dataset
        self.data_path = data_path
        self.get_csv_dataframe()
        top_50_tag_index = self.get_top50_tags()
        file_titles, binary_label = self.write_tags(top_50_tag_index)
        self.split(file_titles, binary_label)


if __name__ == '__main__':
    data_path = sys.argv[1]
    split = Split()
    split.run(data_path)
