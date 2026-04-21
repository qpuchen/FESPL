import argparse
import os

import yaml

from affnet import logger

from affnet.download_utils_base import get_basic_local_path

try:
    from internal.utils.blobby_utils import get_local_path_blobby

    get_local_path = get_local_path_blobby

except ModuleNotFoundError as mnfe:
    get_local_path = get_basic_local_path

import collections
try:
    # Workaround for DeprecationWarning when importing Collections
    collections_abc = collections.abc
except AttributeError:
    collections_abc = collections

DEFAULT_CONFIG_DIR = "config"
from affnet import modeling_arguments


def is_master(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    return node_rank == 0


def flatten_yaml_as_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections_abc.MutableMapping):
            items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_config_file(opts):
    config_file_name = './config/affnet_ade/config.yaml'
    if config_file_name is None:
        return opts
    is_master_node = is_master(opts)

    if is_master_node:
        config_file_name = get_local_path(opts=opts, path=config_file_name)

    if not os.path.isfile(config_file_name):
        if len(config_file_name.split("/")) == 1:
            # loading files from default config folder
            new_config_file_name = "{}/{}".format(DEFAULT_CONFIG_DIR, config_file_name)
            if not os.path.isfile(new_config_file_name) and is_master_node:
                logger.error(
                    "Configuration file neither exists at {} nor at {}".format(
                        config_file_name, new_config_file_name
                    )
                )
            else:
                config_file_name = new_config_file_name
        else:
            # If absolute path of the file is passed
            if not os.path.isfile(config_file_name) and is_master_node:
                logger.error(
                    "Configuration file does not exists at {}".format(config_file_name)
                )

    setattr(opts, "affnet.common.config_file", config_file_name)
    with open(config_file_name, "r") as yaml_file:
        try:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
        except yaml.YAMLError as exc:
            if is_master_node:
                logger.error(
                    "Error while loading config file: {}. Error message: {}".format(
                        config_file_name, str(exc)
                    )
                )

    # override arguments
    override_args = getattr(opts, "override_args", None)
    if override_args is not None:
        for override_k, override_v in override_args.items():
            if hasattr(opts, override_k):
                setattr(opts, override_k, override_v)

    return opts


def parser_to_opts(parser: argparse.ArgumentParser):
    # parse args
    opts = parser.parse_args()
    opts = load_config_file(opts)
    return opts


def get_training_arguments(parse_args=True):
    parser = argparse.ArgumentParser(description="Training arguments", add_help=True)

    # cvnet arguments, including models
    parser = modeling_arguments(parser=parser)

    # wandb
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    # parser.set_defaults(log_wandb=True)
    parser.add_argument('--experiment', default='debug', type=str, metavar='NAME',
                        help='name of train experiment, name of sub-folder for output')


    if parse_args:
        return parser_to_opts(parser)
    else:
        return parser
