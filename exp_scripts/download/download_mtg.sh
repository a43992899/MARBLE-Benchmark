# export PROJECT_ROOT=~/MARBLE-Benchmark

cd $PROJECT_ROOT
mkdir -p data/MTG
mkdir data/MTG/audio-low
cd data/MTG
git clone https://github.com/MTG/mtg-jamendo-dataset.git
pip install -r mtg-jamendo-dataset/scripts/requirements.txt
cd mtg-jamendo-dataset
python3 scripts/download/download.py --dataset raw_30s --type audio-low ../audio-low --unpack --remove

cd $PROJECT_ROOT
