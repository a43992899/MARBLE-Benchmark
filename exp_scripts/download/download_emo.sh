# export PROJECT_ROOT=~/MARBLE-Benchmark

cd $PROJECT_ROOT
mkdir -p data/EMO
cd data/EMO

wget http://cvml.unige.ch/databases/emoMusic/clips_45sec.tar.gz
wget http://cvml.unige.ch/databases/emoMusic/annotations.tar.gz
wget http://cvml.unige.ch/databases/emoMusic/dataset_manual.pdf

tar -xvf clips_45sec.tar.gz
tar -xvf annotations.tar.gz

cd $PROJECT_ROOT
