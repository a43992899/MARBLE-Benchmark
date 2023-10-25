# export PROJECT_ROOT=~/MIR-Benchmark

cd $PROJECT_ROOT
cd data/GS
unzip key-annotations.zip
unzip mtg-key-annotations.zip
cd $PROJECT_ROOT
python benchmark/tasks/GS/preprocess.py --dataset_dir data/GS
