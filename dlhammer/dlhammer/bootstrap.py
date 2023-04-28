# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import sys
import logging

from .logger import bootstrap_logger, logger
from .argparser import bootstrap_args, CONFIG
from .utils.misc import to_string

__all__ = ['bootstrap', 'logger', 'CONFIG']


def bootstrap(default_cfg=None, print_cfg=True):
    """TODO: Docstring for bootstrap.

    Kwargs:
        use_argparser (TODO): TODO
        use_logger (TODO): TODO

    Returns: TODO

    """
    config = bootstrap_args(default_cfg)
    if print_cfg:
        logger.info(to_string(config))
    return config
