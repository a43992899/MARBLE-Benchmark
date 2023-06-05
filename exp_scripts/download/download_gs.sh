# export PROJECT_ROOT=~/MIR-Benchmark

cd $PROJECT_ROOT

mkdir -p data/GS
cd data/GS
wget https://github.com/GiantSteps/giantsteps-mtg-key-dataset/archive/fd7b8c584f7bd6d720d170c325a6d42c9bf75a6b.zip -O mtg-key-annotations.zip
wget https://github.com/GiantSteps/giantsteps-key-dataset/archive/c8cb8aad2cb53f165be51ea099d0dc75c64a844f.zip -O key-annotations.zip

cd $PROJECT_ROOT
python benchmark/tasks/GS/download.py --data_path ./data/GS


