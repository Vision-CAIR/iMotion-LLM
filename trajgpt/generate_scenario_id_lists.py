import glob
import sys
sys.path.append("..")
import argparse
from multiprocessing import Pool
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
import os
import pickle
from mm_viz import *
import time
from scipy.interpolate import interp1d
from matplotlib.patches import Circle
import json
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

                    

#   <internal_waymo_dataset_root>/valid_scenario_ids

data_files = glob.glob('<internal_waymo_dataset_root>/validation_interactive/*')
scenario_ids = {}
for data_file in tqdm(data_files):
    dataset = tf.data.TFRecordDataset(data_file)
    scenario_ids[data_file.split('/')[-1]] = []
    for data in dataset:
        parsed_data = scenario_pb2.Scenario()
        parsed_data.ParseFromString(data.numpy())
        scenario_id = parsed_data.scenario_id
        scenario_ids[data_file.split('/')[-1]].append(scenario_id)

with open('scenario_id_map.json', 'w') as file:
    json.dump(scenario_ids, file, indent=4)  # indent=4 for pretty printing
