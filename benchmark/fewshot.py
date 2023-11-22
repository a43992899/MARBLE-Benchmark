from tqdm import tqdm
import random
import argparse

import wandb
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import pytorch_lightning as pl

import benchmark as bench
from benchmark.utils.config_utils import load_config, override_config, print_config, merge_args_to_config
from benchmark.utils.get_dataloader import get_feature_datasets
from benchmark.utils.get_logger import get_logger
from benchmark.utils.get_callbacks import get_callbacks 


def normalize(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    x /= x.norm(dim=-1, keepdim=True)
    return x

def get_class_centroids(class_data, num_shots):
    class_centroids = dict()
    for label, features in class_data.items():
        # turn list of tensor features into tensor
        features = torch.stack(features)
        # sample num_shots features
        if features.shape[0] > num_shots:
            features = features[random.sample(range(features.shape[0]), num_shots)]
        features /= features.norm(dim=-1, keepdim=True)
        class_centroids[label] = features.mean(dim=0, keepdim=True)
    return class_centroids

def select_downstream_cfg(cfg):
    ret_dict = dict()
    downstream_structure_components = cfg.model.downstream_structure.components
    for component in downstream_structure_components:
        if component.name == 'feature_selector':
            ret_dict['layer'] = component.layer
    return ret_dict

def get_sample_i(dataset, i):
    feature, label, audio_path = dataset[i]
    assert feature.dim() == 3, "feature must be 3D tensor, (num_layers, num_frames, feature_dim)"
    feature = feature.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
    feature = feature.reshape(-1)
    label = label.item()
    return feature, label, audio_path

def main(args):
    cfg = load_config(args.config, namespace=True)
    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, cfg)
    cfg = merge_args_to_config(args, cfg)
    print_config(cfg)

    cfg._runtime = argparse.Namespace() # runtime info

    # force overwrite, since we are using the same config as probe
    # TODO: setup a new config for fewshot
    cfg.trainer.paradigm = 'fewshot'
    cfg.logger.wandb_proj_name = None

    pl.seed_everything(cfg.trainer.seed)

    logger = get_logger(cfg)

    (train_dataset, _, _), \
    (valid_dataset, _, _), \
    (test_dataset, _, _)  = get_feature_datasets(cfg, return_audio_path=True)

    class_data = dict()

    selected_dict = select_downstream_cfg(cfg)

    for i in range(len(train_dataset)):
        feature, label, audio_path = get_sample_i(train_dataset, i)
        if label not in class_data:
            class_data[label] = []
        class_data[label].append(feature)
    
    all_acc = []
    repeat_times = args.iter
    num_shots = args.n_shot
    # repeat the experiment for many times, since different centroid initialization will lead to different results
    for _ in tqdm(range(repeat_times)):
        # compute class centroids
        class_centroids = get_class_centroids(class_data, num_shots)
        class_centroids = torch.cat([class_centroids[i] for i in range(10)])
        class_centroids = normalize(class_centroids)
        results, labels, paths = [], [], []
        for i in range(len(test_dataset)):
            feature, label, audio_path = get_sample_i(test_dataset, i)
            feature = normalize(feature)
            probs = (feature @ class_centroids.T).softmax(dim=-1)
            top_prob, top_label = probs.topk(1, dim=-1)
            top_label = top_label.item()
            results.append(top_label)
            labels.append(label)
            paths.append(audio_path)
        all_acc.append(accuracy_score(labels, results))
    all_acc = np.array(all_acc)
    print(f"Accuracy: {all_acc.mean():.4f} +- {all_acc.std():.4f}")

    # TODO: wandb log
    logger.log_metrics({'fewshot_acc': all_acc.mean(), 'fewshot_acc_std': all_acc.std()})

    wandb.finish()

