import collections
import pprint
from pathlib import Path
from copy import deepcopy

import argparse
import yaml

###################################
# ArgumentParser Helper Functions #
###################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_config_and_override_args(parser):
    """Add arguments for config and override.
    """
    parser.add_argument(
        '-c',
        '--config',
        type=Path,
        required=True,
        help='path to the config file.',
    )
    parser.add_argument(
        '-o',
        '--override',
        type=str,
        required=False,
        default=None,
        help='override specific config values with a string of doulbe comma separated key=value pairs',
    )
    return parser


def add_extract_args(parser):
    """Add arguments for feature extraction subparser.
    """
    parser = add_config_and_override_args(parser)
    parser.add_argument(
        '-r', # rank
        '--shard_rank',
        type=int,
        default=0,
        required=False,
        help='current process rank for distributed extraction.',
    )
    parser.add_argument(
        '-n', # number of shards
        '--n_shard',
        type=int,
        default=1,
        required=False,
        help='total number of processes for distributed extraction.',
    )
    return parser


def add_probe_args(parser):
    """Add arguments for probing subparser.
    """
    parser = add_config_and_override_args(parser)
    return parser


def add_finetune_args(parser):
    """Add arguments for finetuning subparser.
    """
    parser = add_config_and_override_args(parser)
    return parser
