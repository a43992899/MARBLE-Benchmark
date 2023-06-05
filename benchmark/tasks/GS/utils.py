from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import sklearn
import mir_eval
from scipy.stats import mode as scipy_mode

def predict_result_ensemble(outputs_with_meta_ids, device):
    '''
    adapted from jukemir
    input: 
        outputs_with_meta_ids = [(), ...] <= provided by test_step
    return:
        comparisions = [(strategy_name, predicted_label, label), ...] <= mir_eval format
        (torchmetrics.Accuracy support both predicted_label and predicted_logits input)
    '''
    # meta info for GS dataset
    # id_to_label = DATASET_TO_ATTRS["giantsteps_clips"]["labels"]
    classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
    class2id = {c: i for i, c in enumerate(classes)}
    id2class = {v: k for k, v in class2id.items()}
    id_to_label = id2class

    # retrieve all the predicitions and labels
    y = []
    clip_logits = []
    y_meta_id = []
    class_info = []
    for label, pred_logit, meta_id, class_in_str in outputs_with_meta_ids:
        y.append(label)
        clip_logits.append(pred_logit)
        y_meta_id.append(meta_id)
        class_info += list(class_in_str)

    y_meta_id = np.hstack(y_meta_id) # -> (2406,)
    clip_labels = np.hstack(y) # -> (2406, )
    clip_logits =  np.vstack(clip_logits) # -> (2406, 24)
    clip_preds = np.argmax(clip_logits, axis=1)
    class_info = np.array(class_info)
    with torch.no_grad():
        clip_probs = (
            F.softmax(torch.tensor(clip_logits, device=device), dim=-1)
            .cpu()
            .numpy()
        )
    # merge the predictions and labels according to their meta audio id
    # largely adapted from jukemir
    song_uid_to_clip_idxs = defaultdict(list) # meta_id -> actual index in clip_labels&clip_logits
    song_uid_to_label = {}
    for clip_idx, (song_uid, label) in enumerate(zip(y_meta_id, clip_labels)):
        song_uid_to_clip_idxs[song_uid].append(clip_idx)
        if song_uid in song_uid_to_label:
            assert song_uid_to_label[song_uid] == label
        song_uid_to_label[song_uid] = label
    song_uids = sorted(song_uid_to_clip_idxs.keys())
    song_labels = np.array(
        [song_uid_to_label[song_uid] for song_uid in song_uids]
    )
    # Ensemble predictions
    ensemble_strategy_to_song_preds = defaultdict(list)
    for song_uid in song_uids:
        clip_idxs = song_uid_to_clip_idxs[song_uid]

        song_clip_logits = clip_logits[clip_idxs]
        song_clip_preds = clip_preds[clip_idxs]
        song_clip_probs = clip_probs[clip_idxs]
        ensemble_strategy_to_song_preds["vote"].append(
            # scipy_mode(song_clip_preds, keepdims=True).mode[0]
            scipy_mode(song_clip_preds).mode[0] # , keepdims=True
        )
        ensemble_strategy_to_song_preds["max"].append(
            song_clip_logits.max(axis=0).argmax()
        )
        ensemble_strategy_to_song_preds["gmean"].append(
            song_clip_logits.mean(axis=0).argmax()
        )
        ensemble_strategy_to_song_preds["mean"].append(
            song_clip_probs.mean(axis=0).argmax()
        )

    # Compute all metrics
    comparisons = [
        (
            "clip",
            np.argmax(clip_probs, axis=1),
            clip_labels,
        )
    ]
    comparisons += [
        (f"ensemble_{strategy_name}", np.array(strategy_preds), song_labels)
        for strategy_name, strategy_preds in ensemble_strategy_to_song_preds.items()
    ]

    # return comparisons
    def _compute_accuracy_and_scores(preds, labels):
        assert preds.shape == labels.shape
        correct = preds == labels
        accuracy = correct.astype(np.float32).mean()
        # print(class_info)
        # print(labels)
        # print(preds.shape)
        # print(preds[0])
        scores = [
            mir_eval.key.weighted_score(
                id_to_label[ref_key], id_to_label[est_key]
            )
            for ref_key, est_key in zip(labels, preds)
        ]
        # scores = [
        #     mir_eval.key.weighted_score(
        #         class_info[ref_key], class_info[est_key]
        #     )
        #     for ref_key, est_key in zip(labels, preds)
        # ]
        return accuracy, np.mean(scores)
    
    log_dict = {}
    for prefix, preds, labels in comparisons:
        accuracy, score = _compute_accuracy_and_scores(preds, labels)
        log_dict [f"{prefix}_accuracy"] = accuracy
        log_dict [f"{prefix}_score"] = score

    best_strategy_name = None
    best_score = float("-inf")

    for strategy_name in ensemble_strategy_to_song_preds.keys():
        score = log_dict[f"ensemble_{strategy_name}_score"]
        if score > best_score:
            best_strategy_name = strategy_name
            best_score = score
    log_dict[f"best_ensemble_accuracy"] = log_dict[f"ensemble_{best_strategy_name}_accuracy"]
    log_dict[f"best_ensemble_score"] = log_dict[f"ensemble_{best_strategy_name}_score"]
    log_dict["best_ensemble_strategy"] = best_strategy_name
    
    return log_dict