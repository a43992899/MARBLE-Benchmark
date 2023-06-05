import benchmark as bench
from benchmark.utils.config_utils import search_enumerate

from benchmark.tasks.MTT.MTT_dataset import FeatureDataset as MTTFeatureDataset, AudioDataset as MTTAudioDataset
from benchmark.tasks.GTZAN.GTZAN_dataset import FeatureDataset as GTZANFeatureDataset, AudioDataset as GTZANAudioDataset
from benchmark.tasks.VocalSet.VocalSetS_dataset import FeatureDataset as VocalSetSFeatureDataset, AudioDataset as VocalSetSAudioDataset
from benchmark.tasks.VocalSet.VocalSetT_dataset import FeatureDataset as VocalSetTFeatureDataset, AudioDataset as VocalSetTAudioDataset
from benchmark.tasks.GS.GS_dataset import FeatureDataset as GSFeatureDataset, AudioDataset as GSAudioDataset
from benchmark.tasks.EMO.EMO_dataset import FeatureDataset as EMOFeatureDataset, AudioDataset as EMOAudioDataset
from benchmark.tasks.MAESTRO.MAESTRO_dataset import MaestroAudioDataset
from benchmark.tasks.GTZAN.GTZANBT_dataset import AudioDataset as GTZANBTAudioDataset
from benchmark.tasks.NSynth.NSynthI_dataset import FeatureDataset as NSynthIFeatureDataset, AudioDataset as NSynthIAudioDataset
from benchmark.tasks.NSynth.NSynthP_dataset import FeatureDataset as NSynthPFeatureDataset, AudioDataset as NSynthPAudioDataset
from benchmark.tasks.MTG.MTGGenre_dataset import FeatureDataset as MTGGenreFeatureDataset # , AudioDataset as MTGGenreAudioDataset
from benchmark.tasks.MTG.MTGInstrument_dataset import FeatureDataset as MTGInstrumentFeatureDataset # , AudioDataset as MTGInstrumentAudioDataset
from benchmark.tasks.MTG.MTGMood_dataset import FeatureDataset as MTGMoodFeatureDataset # , AudioDataset as MTGMoodAudioDataset
from benchmark.tasks.MTG.MTGTop50_dataset import FeatureDataset as MTGTop50FeatureDataset # , AudioDataset as MTGTop50AudioDataset
from benchmark.tasks.MUSDB18.MUSDB18_dataset import FixedSourcesTrackFolderDataset, aug_from_str
from torch.utils.data import DataLoader, Dataset
import torch


def get_audio_datasets(args):
    train_sampler, valid_sampler, test_sampler = None, None, None
    train_collate_fn, valid_collate_fn, test_collate_fn = None, None, None

    if args.dataset == "MUSDB18":
        dataset_kwargs = {
            "root": args.audio_dir,
            "target_file": args.target,
            "sample_rate": args.target_sr,
        }
        source_augmentations = aug_from_str(["gain", "channelswap"])
        train_dataset = FixedSourcesTrackFolderDataset(split='train', seq_duration=6, source_augmentations=source_augmentations, random_chunks=True, random_track_mix=True, **dataset_kwargs)
        valid_dataset = FixedSourcesTrackFolderDataset(split='valid', seq_duration=None, samples_per_track=1, **dataset_kwargs)
        test_dataset = FixedSourcesTrackFolderDataset(split='test', seq_duration=None, samples_per_track=1, **dataset_kwargs)
    else:
        Task_Dataset = eval(f"{args.dataset.dataset}AudioDataset")
        train_dataset = Task_Dataset(args, split="train")
        valid_dataset = Task_Dataset(args, split="valid")
        test_dataset = Task_Dataset(args, split="test")

    return (
        (train_dataset, train_sampler, train_collate_fn), 
        (valid_dataset, valid_sampler, valid_collate_fn), 
        (test_dataset, test_sampler, test_collate_fn)
    )


def get_feature_datasets(args):
    Task_Dataset = eval(f"{args.dataset.dataset}FeatureDataset")
    layer = search_enumerate(
        args.model.downstream_structure.components, 
        name="feature_selector", 
        key="layer"
    )
    train_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir, 
        metadata_dir=args.dataset.metadata_dir, 
        split="train", 
        layer=layer
    )
    valid_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir, 
        metadata_dir=args.dataset.metadata_dir, 
        split="valid", 
        layer=layer
    )
    test_dataset = Task_Dataset(
        feature_dir=args.dataset.input_dir, 
        metadata_dir=args.dataset.metadata_dir, 
        split="test", 
        layer=layer
    )

    train_collate_fn = train_dataset.train_collate_fn
    valid_collate_fn = valid_dataset.test_collate_fn
    test_collate_fn = test_dataset.test_collate_fn

    train_sampler, valid_sampler, test_sampler = None, None, None

    return (
        (train_dataset, train_sampler, train_collate_fn), 
        (valid_dataset, valid_sampler, valid_collate_fn), 
        (test_dataset, test_sampler, test_collate_fn)
    )


dataset_functions = {
    'feature': get_feature_datasets,
    'audio': get_audio_datasets
}


def get_dataloaders(args):
    dataset_type = args.dataset.input_type

    if dataset_type in dataset_functions:
        (train_dataset, train_sampler, train_collate_fn), \
        (valid_dataset, valid_sampler, valid_collate_fn), \
        (test_dataset, test_sampler, test_collate_fn) = dataset_functions[dataset_type](args)
    else:
        raise NotImplementedError(f"get_dataloaders() of dataset type {dataset_type} not implemented")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.dataloader.batch_size.train, 
        shuffle=True, 
        num_workers=args.dataloader.num_workers, 
        collate_fn=train_collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.dataloader.batch_size.valid, 
        shuffle=False, 
        num_workers=args.dataloader.num_workers, 
        collate_fn=test_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.dataloader.batch_size.test, 
        shuffle=False, 
        num_workers=args.dataloader.num_workers, 
        collate_fn=test_collate_fn
    )

    return train_loader, valid_loader, test_loader

