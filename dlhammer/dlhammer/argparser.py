# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import os
import argparse
import datetime
from functools import partial
import yaml
from easydict import EasyDict

# from .utils import get_vacant_gpu
from .logger import bootstrap_logger, logger
from .utils.system import get_available_gpuids
from .utils.misc import merge_dict, merge_opts, to_string, eval_dict_leaf

CONFIG = EasyDict()

BASE_CONFIG = {
    'OUTPUT_DIR': './workspace',
    'SESSION': 'base',
    'NUM_GPUS': 1,
    'LOG_NAME': 'log.txt'
}


def bootstrap_args(default_params=None):
    """get the params from yaml file and args. The args will override arguemnts in the yaml file.
    Returns: EasyDict instance.

    """
    parser = define_default_arg_parser()
    cfg = update_config(parser, default_params)
    create_workspace(cfg)    #create workspace

    CONFIG.update(cfg)
    bootstrap_logger(get_logfile(CONFIG))    # setup logger
    setup_gpu(CONFIG.NUM_GPUS)    #setup gpu

    return cfg


def setup_gpu(ngpu):
    gpuids = get_available_gpuids()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpuids[:ngpu]])


def get_logfile(config):
    return os.path.join(config.WORKSPACE, config.LOG_NAME)


def define_default_arg_parser():
    """Define a default arg_parser.

    Returns: 
        A argparse.ArgumentParser. More arguments can be added.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='load configs from yaml file', default='', type=str)
    parser.add_argument('opts',
                        default=None,
                        nargs='*',
                        help='modify config options using the command-line')

    return parser


def update_config(arg_parser, default_config=None):
    """ update argparser to args.

    Args:
        arg_parser: argparse.ArgumentParser.
    """

    parsed, unknown = arg_parser.parse_known_args()
    if default_config and parsed.cfg == "" and "cfg" in default_config:
        parsed.cfg = default_config["cfg"]

    config = EasyDict(BASE_CONFIG.copy())
    config['cfg'] = parsed.cfg
    # update default config
    if default_config is not None:
        config.update(default_config)

    # merge config from yaml
    if os.path.isfile(config.cfg):
        with open(config.cfg, 'r') as f:
            yml_config = yaml.full_load(f)
        config = merge_dict(config, yml_config)

    # merge opts
    config = merge_opts(config, parsed.opts)

    # eval values
    config = eval_dict_leaf(config)

    return config


def create_workspace(cfg):
    cfg_name, ext = os.path.splitext(os.path.basename(cfg.cfg))
    workspace = os.path.join(cfg.OUTPUT_DIR, cfg_name, cfg.SESSION)
    os.makedirs(workspace, exist_ok=True)
    cfg.WORKSPACE = workspace
