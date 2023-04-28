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

CURRENT_FILE_DIRECTORY = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CURRENT_FILE_DIRECTORY, '../..'))
sys.path.append(os.path.join(CURRENT_FILE_DIRECTORY, '.'))

from dlhammer import bootstrap, logger
bootstrap()

logger.info('dummy output')

raise Exception('dummy error')
