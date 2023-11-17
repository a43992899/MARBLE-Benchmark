import wandb
import argparse
import torch
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
        features = torch.tensor(np.array(features))
        # sample num_shots features
        if features.shape[0] > num_shots:
            features = features[random.sample(range(features.shape[0]), num_shots)]
        features /= features.norm(dim=-1, keepdim=True)
        class_centroids[label] = features.mean(dim=0, keepdim=True)
    return class_centroids

def main(args):
    cfg = load_config(args.config, namespace=True)
    if args.override is not None and args.override.lower() != "none":
        override_config(args.override, cfg)
    cfg = merge_args_to_config(args, cfg)
    print_config(cfg)

    cfg._runtime = argparse.Namespace() # runtime info

    assert cfg.trainer.paradigm == 'probe', "paradigm must be probe for probe.py"
    pl.seed_everything(cfg.trainer.seed)

    logger = get_logger(cfg)

    (train_dataset, _, _), \
    (valid_dataset, _, _), \
    (test_dataset, _, _)  = get_feature_datasets(cfg)

    class_data = dict()

    for i in range(len(train_dataset)):
        feature, label, audio_path = train_dataset[i]
        # TODO: 重写
        if feature.shape[0] != 768:  # cat all layers to one vector
            feature = feature.reshape(-1)
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
        for feature, label, audio_path in test_dataset:
            if feature.shape[0] != 768:  # cat all layers to one vector
                feature = feature.reshape(-1)
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

    wandb.finish()

