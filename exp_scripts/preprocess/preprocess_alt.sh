# To prepare MulJam dataset for the lyrics transcription task
# download the full MTG first: ./exp_scripts/download/download_mtg.sh

wget -O benchmark/tasks/ALT/metadata/test.meta https://raw.githubusercontent.com/zhuole1025/LyricWhiz/main/MulJam_v2.0/preconstructed-split/test.meta
wget -O benchmark/tasks/ALT/metadata/valid.meta https://raw.githubusercontent.com/zhuole1025/LyricWhiz/main/MulJam_v2.0/preconstructed-split/valid.meta
wget -O benchmark/tasks/ALT/metadata/train.meta https://raw.githubusercontent.com/zhuole1025/LyricWhiz/main/MulJam_v2.0/preconstructed-split/train.meta

python benchmark/tasks/ALT/preprocess.py --dataset_dir ./data/ALT