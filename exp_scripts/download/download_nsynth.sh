# export PROJECT_ROOT=~/MIR-Benchmark

cd $PROJECT_ROOT

mkdir -p data/NSynth
cd data/NSynth

wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

tar -xvf nsynth-train.jsonwav.tar.gz
tar -xvf nsynth-valid.jsonwav.tar.gz
tar -xvf nsynth-test.jsonwav.tar.gz

cd $PROJECT_ROOT