# export PROJECT_ROOT=~/MIR-Benchmark

cd $PROJECT_ROOT
mkdir -p data/MTT
cd data/MTT

wget "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001"
wget "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002"
wget "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003"
cat mp3.zip* > mp3.zip
mkdir ./mp3
unzip mp3.zip -d ./mp3
rm mp3.*

wget https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv
## below are not used, you can decide to download them or not
# wget http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.csv
# wget http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.sql.zip
# wget http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3_echonest_xml.zip
# wget http://mi.soi.city.ac.uk/datasets/magnatagatune/comparisons_final.csv


cd $PROJECT_ROOT