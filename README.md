# MARBLE: Music Audio Representation Benchmark for Universal Evaluation

Currently support:
- Linear probing pretrained features.
- Various MIR datasets:
  - `MTT`: magnatagatune, multilabel cls
  - `MTG`: MTG-jamendo, multilabel cls
  - `GTZAN`: GTZAN, multiclass cls
  - `GS`: giantsteps, multiclass cls 
  - `EMO`: emomusic, regression
  - `VocalSet`: VocalSet, multiclass cls

TODOs:
- Support `GTZANBT`: GTZAN Beat Tracking, will be updated soon.
- Support `MUSDB18`: MUSDB18, source separation, will be updated soon.
- Support traditional handcrafted features.
- Support `MAESTRO`: maestro, piano transcription
- Support lyrics transcription.
- Support few-shot inference.

## Getting Start
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

## Paper:
```code
@article{yuan2023marble,
  title={MARBLE: Music Audio Representation Benchmark for Universal Evaluation},
  author={Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Zhuo, Le and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and others},
  journal={arXiv preprint arXiv:2306.10548},
  year={2023}
}
```
