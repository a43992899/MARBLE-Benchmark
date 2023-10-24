# export PROJECT_ROOT=~/MARBLE-Benchmark

cd $PROJECT_ROOT
mkdir -p data/GTZAN
cd data/GTZAN
## wget http://opihi.cs.uvic.ca/sound/genres.tar.gz # this link is dead as of 2023-02-18

# Check if ~/.kaggle/kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Error: ~/.kaggle/kaggle.json does not exist!"
    echo "Please create your kaggle.json file from https://www.kaggle.com/<username>/account and place it in the current directory, then run this script."
    exit 1
fi

pip install kaggle

## create your kaggle.json file from https://www.kaggle.com/<username>/account, place it in the current directory, and run this script
mkdir ~/.kaggle
cp ./kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip gtzan-dataset-music-genre-classification.zip

ln -s Data/genres_original ./genres

wget https://raw.githubusercontent.com/coreyker/dnn-mgr/bdad579ea6cb37b665ea6019fe1026a6ce20cbc7/gtzan/train_filtered.txt
# jazz.00054.wav is corrupted in kaggle source, remove it from train_filtered.txt
sed -e '/jazz.00054.wav/d' train_filtered.txt > _train_filtered.txt
mv _train_filtered.txt train_filtered.txt
wget https://raw.githubusercontent.com/coreyker/dnn-mgr/bdad579ea6cb37b665ea6019fe1026a6ce20cbc7/gtzan/valid_filtered.txt
wget https://raw.githubusercontent.com/coreyker/dnn-mgr/bdad579ea6cb37b665ea6019fe1026a6ce20cbc7/gtzan/test_filtered.txt

# beat tracking label
wget http://anasynth.ircam.fr/home/system/files/attachment_uploads/marchand/private/GTZAN-Rhythm_v2_ismir2015_lbd_2015-10-28.tar_.gz

tar xzvf GTZAN-Rhythm_v2_ismir2015_lbd_2015-10-28.tar_.gz

cd $PROJECT_ROOT