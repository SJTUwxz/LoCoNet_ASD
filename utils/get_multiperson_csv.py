import os, pandas
import json

phase = "test"
path = "/nfs/jolteon/data/ssd/xiziwang/AVA_dataset/csv"

if phase == "train":
    csv_f = "train_loader.csv"
    csv_orig = "train_orig.csv"
elif phase == "val":
    csv_f = "val_loader.csv"
    csv_orig = "val_orig.csv"
else:
    csv_f = "test_loader.csv"
    csv_orig = "test_orig.csv"

orig_df = pandas.read_csv(os.path.join(path, csv_orig))
entity_data = {}
ts_to_entity = {}

for index, row in orig_df.iterrows():

    entity_id = row['entity_id']
    video_id = row['video_id']
    if row['label'] == "SPEAKING_AUDIBLE":
        label = 1
    else:
        label = 0
    ts = float(row['frame_timestamp'])
    if video_id not in entity_data.keys():
        entity_data[video_id] = {}
    if entity_id not in entity_data[video_id].keys():
        entity_data[video_id][entity_id] = {}
    if ts not in entity_data[video_id][entity_id].keys():
        entity_data[video_id][entity_id][ts] = []

    entity_data[video_id][entity_id][ts] = label

    if video_id not in ts_to_entity.keys():
        ts_to_entity[video_id] = {}
    if ts not in ts_to_entity[video_id].keys():
        ts_to_entity[video_id][ts] = []
    ts_to_entity[video_id][ts].append(entity_id)

with open(os.path.join(path, phase + "_entity.json"), 'w') as f:
    json.dump(entity_data, f)

with open(os.path.join(path, phase + "_ts.json"), 'w') as f:
    json.dump(ts_to_entity, f)
