# export PROJECT_ROOT=~/MIR-Benchmark

cd $PROJECT_ROOT
mkdir -p data/maestro
cd data/maestro

wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip
unzip maestro-v2.0.0.zip
rm maestro-v2.0.0.zip

cd $PROJECT_ROOT