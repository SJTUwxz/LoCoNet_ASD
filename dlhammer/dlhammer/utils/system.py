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
import subprocess
import numpy as np


def get_available_gpuids():
    """
    Returns: the gpu ids sorted in descending order w.r.t occupied memory.
    """
    com = "nvidia-smi|sed -n '/%/p'|sed 's/|/\\n/g'|sed -n '/MiB/p'|sed 's/ //g'|sed 's/MiB/\\n/'|sed '/\\//d'"
    gpum = subprocess.check_output(com, shell=True)
    gpum = gpum.decode('utf-8').split('\n')
    gpum = gpum[:-1]
    sorted_gpuid = np.argsort(gpum)
    return sorted_gpuid
