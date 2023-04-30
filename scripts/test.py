# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================
import os
from multiprocessing import Pool

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def evaluate_epoch(epoch):
    # model_path = f"/nfs/joltik/data/ssd/xiziwang/TalkNet_models/baseline/1.a.pl/model/{epoch:04d}.pth"
    config_name = "multi"
    output_name = "replicate"
    attention = "speaker_temporal"
    numspeakers = 3
    av_layers = 3
    adjust_attention = 0
    model_path = f"/nfs/joltik/data/ssd/xiziwang/TalkNet_models/{output_name}/model/model_{epoch:04d}.model"
    cmd = f"python -W ignore::UserWarning test_multicard.py --cfg configs/{config_name}.yaml  RESUME_PATH {model_path} RESUME_EPOCH {epoch} MODEL.AV {attention} MODEL.NUM_SPEAKERS {numspeakers} MODEL.AV_layers {av_layers} MODEL.ADJUST_ATTENTION {adjust_attention}"
    os.system(cmd)


pool = Pool(1)
epochs = list(range(17, 26))
pool.map(evaluate_epoch, epochs)
