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
- Fix config files of mule, music2vec, musicnn, handcrafted, clmr
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

## Development Workflow

### Branching Strategy
- **`main` branch**: This branch contains the stable version of the project. All new features and fixes should be developed on separate branches and merged into `main` once they are tested and reviewed.
  
- **`dev` branch**: This is the primary development branch where all the features and bug fixes are merged. When the `dev` branch is stable and ready for a new version, it will be merged into the `main` branch.

- **Feature Branches**: For each new feature or improvement, create a new branch from `dev`. The naming convention is `feature/descriptive-name`, e.g., `feature/add-GTZANBT-support`.

- **Bugfix Branches**: If there's a bug to fix, create a branch from `dev`. Name it `bugfix/descriptive-name`, e.g., `bugfix/fix-data-loading-issue`.

## Paper:
```code
@article{yuan2023marble,
  title={MARBLE: Music Audio Representation Benchmark for Universal Evaluation},
  author={Yuan, Ruibin and Ma, Yinghao and Li, Yizhi and Zhang, Ge and Chen, Xingran and Yin, Hanzhi and Zhuo, Le and Liu, Yiqi and Huang, Jiawen and Tian, Zeyue and others},
  journal={arXiv preprint arXiv:2306.10548},
  year={2023}
}
```
