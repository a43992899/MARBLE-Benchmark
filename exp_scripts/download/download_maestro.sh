# export PROJECT_ROOT=~/MARBLE-Benchmark

cd $PROJECT_ROOT
mkdir -p data/MAESTRO
cd data/MAESTRO

wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip
unzip maestro-v2.0.0.zip
rm maestro-v2.0.0.zip

cd $PROJECT_ROOT