r"""Compute active speaker detection performance for the AVA dataset.
Please send any questions about this code to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users
Example usage:
python -O get_ava_active_speaker_performance.py \
-g testdata/eval.csv \
-p testdata/predictions.csv \
-v
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def parse_arguments():
    """Parses command-line flags.
  Returns:
    args: a named tuple containing three file objects args.labelmap,
    args.groundtruth, and args.detections.
  """
    parser = argparse.ArgumentParser()
    parser.add_argument("-g",
                        "--groundtruth",
                        help="CSV file containing ground truth.",
                        type=argparse.FileType("r"),
                        required=True)
    parser.add_argument("-p",
                        "--predictions",
                        help="CSV file containing active speaker predictions.",
                        type=argparse.FileType("r"),
                        required=True)
    parser.add_argument("-v", "--verbose", help="Increase output verbosity.", action="store_true")
    return parser.parse_args()


def run_evaluation(groundtruth, predictions):
    prediction = pd.read_csv(predictions)
    groundtruth = pd.read_csv(groundtruth)
    wrong_list = []
    num = 0
    audible_num = 0
    total = 0
    for i, row in prediction.iterrows():
        entity_id = row['entity_id']
        ts = row['frame_timestamp']
        if row['score'] < 0.5:
            label = "NOT_SPEAKING"
        else:
            label = "SPEAKING_AUDIBLE"

        true_label = groundtruth.loc[(groundtruth['entity_id'] == entity_id) &
                                     (groundtruth['frame_timestamp'] == ts)].iloc[0]["label"]
        if true_label != label:
            wrong_list.append([entity_id, ts, true_label, label])

        if label == "SPEAKING_AUDIBLE":
            num += 1
        if true_label == "SPEAKING_AUDIBLE":
            audible_num += 1
        total += 1
    print(num, audible_num, total)

    df = pd.DataFrame(wrong_list, columns=['entity_id', 'frame_timestamp', "gt", "prediction"])
    df = df.sort_values(by=["frame_timestamp"])
    df.to_csv("wrong_list.csv")


def main():
    start = time.time()
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    del args.verbose
    run_evaluation(**vars(args))
    logging.info("Computed in %s seconds", time.time() - start)


if __name__ == "__main__":
    main()
