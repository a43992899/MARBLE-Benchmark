# Benchmark for Downstream MIR Tasks

Currently support:
- Linear probing MAP pretrain/handcrafted features.
- Fewshot inference.
- Various MIR datasets:
  - `MTT`: magnatagatune, multilabel cls
  - `MTG`: MTG-jamendo, multilabel cls
  - `GTZAN`: GTZAN, multiclass cls
  - `GS`: giantsteps, multiclass cls 
  - `EMO`: emomusic, regression
  - `MAESTRO`: maestro, piano transcription
  - `VocalSet`: VocalSet, multiclass cls

TODO:
- Support finetune MAP pretrain model.
- Support time variant tasks, like transcription.

Please first make sure you are already at the ${PROJECT_ROOT} and have activated your virtual environment.
```bash
export PROJECT_ROOT=/path/to/this/project
cd ${PROJECT_ROOT}
conda activate ${YOUR_ENV}
```

## Download & Preprocess
First run the following script to create data dir.
```bash
cd ${PROJECT_ROOT}
mkdir data
mkdir ./data/wandb # wandb log dir, you should create one if you don't have WANDB_LOG_DIR
mkdir ./data/hubert_data # huggingface hubert checkpoints
```
Then download the datasets and preprocess them. Note that you should have `wget` installed. Not all datasets need preprocessing.
```bash
bash exp_scripts/download/download_emo.sh
bash exp_scripts/preprocess/preprocess_emo.sh # You may skip this step for some datasets.
```


## Extract Features 
Simply do the following
```bash
python . extract -c configs/mert/MERT-v1-95M/EMO.yaml
```
If you want to change the settings, run below to see help.
```bash
python . extract -h
```

## Linear Probing
You should do `wandb login` first. Then do
```bash
python . probe -c configs/mert/MERT-v1-95M/EMO.yaml
```
If you want to change the settings, run below to see help.
```bash
python . probe -h
```

## Few-shot Inference
Script is under `benchmark/fewshot.ipynb`. The code is outdated. 
