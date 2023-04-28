# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import os
import sys
import logging

logger = logging.getLogger('DLHammer')


def bootstrap_logger(logfile=None, fmt=None):
    """TODO: Docstring for bootstrap_logger.

    Args:
        logfile (str): file path logging to.

    Kwargs:
        fmt (TODO): TODO

    Returns: TODO

    """
    if fmt is None:
        # fmt = '%(asctime)s - %(levelname)-5s - [%(filename)s:%(lineno)d] %(message)s'
        fmt = '%(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    #log to file
    if logfile is not None:
        formatter = logging.Formatter(fmt)
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # sys.stdout = LoggerWriter(sys.stdout, logger.info)
    # sys.stderr = LoggerWriter(sys.stderr, logger.error)
    return


class LoggerWriter(object):

    def __init__(self, stream, logfct):
        self.terminal = stream
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.rstrip('\n'))

            message = ''.join(self.buf)
            self.logfct(message)

            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass
