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
- For more plans visit [notion page](https://www.notion.so/yizhilll/Performance-Evaluation-Build-MIR-Benchmark-64d23b676ffd46e6a0bacb67862828bc)

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

If you are on yrb-ubuntu, simply do
```bash
ln -s /home/yrb/code/MIR-Benchmark/data ./data
```

(Optional) To download the huggingface model from gdrive
```bash
bash exp_scripts/download_HF_with_rclone.sh
```

## Extract Features 
If you only want to extract MAP pretrain features with the default settings, we recommend you to run the below shell scripts.  
```bash
bash exp_scripts/extract_features_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${DATASET} ${TARGET_SAMPLE_RATE} ${N_SHARD} ${DEVICE_LIST} ${PROCESSOR_NORMALIZE} ${ACCELERATOR}
```
This shell script calls `benchmark/extract_bert_features.py`.  
If you want to change the settings, run below to see help.
```bash
python . extract-hubert-features -h
```

## Linear Probing
You should do `wandb login` first, contact `yizhi` for setting up.  
If you only want to probe hubert features from a huggingface checkpoint with the default settings, we recommend you to run the below shell scripts. Note that you should [run hubert feature extraction](#extract-hubert-features) first.
```bash
bash exp_scripts/probe_by_dataset.sh ${HF_CHECKPOINT_DIR} ${MODEL_TYPE} ${OUTPUT_FEAT_ROOT} ${TASK} ${MODEL_SETTING} ${ACCELERATOR} ${ACC_PRECISION} ${WANDB_OFF}
```
If you want to change the settings, run below to see help.
```bash
python . probe -h
```

## Few-shot Inference
Script is under `benchmark/fewshot.ipynb`. The code is outdated. 
