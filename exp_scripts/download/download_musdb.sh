# export PROJECT_ROOT=~/MARBLE-Benchmark

cd $PROJECT_ROOT
mkdir -p data/musdb
cd data/musdb

wget https://zenodo.org/record/1117372/files/musdb18.zip
unzip musdb18.zip
rm musdb18.zip

musdbconvert musdb18 musdb18_wav
rm -r musdb18
mv musdb18_wav musdb18

cd $PROJECT_ROOT