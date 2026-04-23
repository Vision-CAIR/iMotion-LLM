import sys
sys.path.append(".")
sys.path.append("/home/felembaa/projects/iMotion-LLM-ICLR/")
import os
import torch
import logging
import glob
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
import random
import json
import matplotlib.pyplot as plt
import shutil
from minigpt4.datasets.datasets.traj_dataset import TrajAlignDataset
import time  # Add at the top with other imports
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Function to check and show image with title
# Old implementation commented out
"""
def check_and_show_image(file_name, category, class_label, image_dir, last_shown_scene, last_shown_category):
    image_path = os.path.join(image_dir, file_name + '.png')
    if os.path.exists(image_path):
        class_label_ = ['stationary', 'straight', 'right', 'left', 'left u-turn']
        class_label_ = class_label_[int(class_label)]
        
        should_show_image = (file_name != last_shown_scene[0] or 
                           (last_shown_category == 'gt1' and category in ['pos1', 'neg1']))
        
        if should_show_image:
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title(f"{file_name}", fontsize=14, color='black')
            plt.axis('off')
            plt.savefig('ex_check.png', dpi=300, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
            plt.show()
            last_shown_scene[0] = file_name
            last_shown_category = category
        
        category_indicator = {
            'gt1': 'GT [+]  ',
            'pos1': 'OF [O+]',
            'neg1': 'INF [-]'
        }
        while True:
            user_input = input(f"{category_indicator[category]} {class_label_} (1:True/0:False): ").strip().lower()
            if user_input in ['true', 'false', '0', '1']:
                return user_input == 'true' or user_input == '1'
    else:
        print(f"Image not found: {image_path}")
    return False
"""

# New implementation
def check_and_show_image(file_name, category, class_label, image_dir, last_shown_scene, last_shown_category):
    image_path = os.path.join(image_dir, file_name + '.png')
    if os.path.exists(image_path):
        # Define class label
        class_label_ = ['stationary', 'straight', 'right', 'left', 'left u-turn']
        class_label_ = class_label_[int(class_label)]
        
        # Only copy image if it's a new scene or if we're switching from gt1 to pos1/neg1
        should_show_image = (file_name != last_shown_scene[0] or 
                           (last_shown_category == 'gt1' and category in ['pos1', 'neg1']))
        
        if should_show_image:
            # Copy the image instead of plotting
            shutil.copy2(image_path, 'ex_check.png')
            last_shown_scene[0] = file_name
            last_shown_category = category
        
        # Add category indicator to prompt
        category_indicator = {
            'gt1': 'GT [+]  ',
            'pos1': 'OF [O+]',
            'neg1': 'INF [-]'
        }
        while True:
            user_input = input(f"{category_indicator[category]} {class_label_} (1:True/0:False): ").strip().lower()
            if user_input in ['true', 'false', '0', '1']:
                return user_input == 'true' or user_input == '1'
    else:
        print(f"Image not found: {image_path}")
    return False


def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def update_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_meta_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        # Create an empty JSON file if it does not exist
        with open(file_path, "w") as f:
            json.dump({}, f)
        return {}

def main():
    root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation_more_data/'
    valid_dir = 'validation'
    
    ## generate all meta data of all categories if it does not exist
    filenames = glob.glob(root_dir+valid_dir+'/*')
    dataset_ = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=0, contrastive=False, train=False, two_agent=False, act_json=False, return_meta_data=True, num_classes=5)
    loader = DataLoader(dataset_, batch_size=1,num_workers=0, shuffle=True)
    print(len(dataset_))
    output_dir = root_dir+valid_dir+'_eval_meta/eval_meta.json'
        
    ## generate per category quality checked meta data
    categories = ['gt1', 'pos1', 'neg1']
    desired_per_label_count = [300, 300, 300, 300, 300]
    # Define JSON output directories for each category
    category_output_dirs = {
        'gt1': root_dir + valid_dir + '_eval_meta/meta_gt1.json',
        'pos1': root_dir + valid_dir + '_eval_meta/meta_pos1.json',
        'neg1': root_dir + valid_dir + '_eval_meta/meta_neg1.json'
    }
    
    # Define JSON output directories for rejected examples
    rejected_output_dirs = {
        'gt1': root_dir + valid_dir + '_eval_meta/rejected_gt1.json',
        'pos1': root_dir + valid_dir + '_eval_meta/rejected_pos1.json',
        'neg1': root_dir + valid_dir + '_eval_meta/rejected_neg1.json'
    }
    
    # Load meta data and rejected data for each category
    meta_data = {category: load_meta_data(category_output_dirs[category]) for category in categories}
    rejected_data = {category: load_meta_data(rejected_output_dirs[category]) for category in categories}

    output_root_dir = root_dir + valid_dir
    # image_dir = root_dir + 'validation_fig/'
    image_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/feb16_2025/validation_more_data/validation_fig'

    # Load meta data for each category JSON file
    meta_data = {category: load_meta_data(category_output_dirs[category]) for category in categories}

    # Initialize counters based on existing entries
    counters = {cat: [0] * len(desired_per_label_count) for cat in categories}

    # Count existing entries in each JSON file
    for category in categories:
        for file_name, label_data in meta_data[category].items():
            # Extract the label from the value dictionary
            if isinstance(label_data, dict):
                if category == 'gt1':
                    label = label_data['gt1']
                    counters[category][label] += 1
                else:
                    label = label_data[category]  # pos1 or neg1
                    counters[category][label] += 1

    # Print current count
    for counter_ in counters:
        print(f"{counter_}: {counters[counter_]}")

    # Load dataset
    dataset_ = TrajAlignDataset(root_dir+valid_dir+'/*', act=True, template_select=0, contrastive=False, train=False, two_agent=False, act_json=False, return_meta_data=True, num_classes=5)
    loader = DataLoader(dataset_, batch_size=1, num_workers=0)

    # Initialize last shown scene tracker (using list for mutable reference)
    last_shown_scene = ['']
    last_shown_category = ''

    # Initialize timing variables
    start_time = time.time()
    example_count = 0

    # Process each data sample
    for data in tqdm(loader):
        file_name = data['file_name'][0]
        example_start_time = time.time()
        examples_this_iteration = 0
        
        # Calculate meta information on the go
        meta_info = {
            'gt1': data['act'][0].item(),
            'pos1': list(data['positive_acts'][0].numpy()),
            'neg1': list(data['contrastive_acts'][0].numpy()),
        }

        # Iterate through each category and label
        for category in categories:
            for desired_label, max_count in enumerate(desired_per_label_count):
                if counters[category][desired_label] >= max_count:
                    continue  # Skip if already reached desired count

                # Check for each category and desired_label
                if (category == 'gt1' and meta_info['gt1'] == desired_label) or \
                   (category in ['pos1', 'neg1'] and desired_label in meta_info[category]):
                    # Create new key with act label and check if it exists
                    new_key = f"{file_name}_act{desired_label}"
                    if new_key in meta_data[category] or new_key in rejected_data[category]:
                        continue
                        
                    valid = check_and_show_image(file_name, category, desired_label, image_dir, 
                                               last_shown_scene, last_shown_category)
                    if valid:
                        # Initialize or update entry based on category
                        if category == 'gt1':
                            meta_data[category][new_key] = {'gt1': desired_label}
                        else:
                            # For pos1 and neg1, include both the category label and gt1
                            meta_data[category][new_key] = {
                                category: desired_label,
                                'gt1': meta_info['gt1']
                            }
                        
                        counters[category][desired_label] += 1
                        update_json(category_output_dirs[category], convert_to_serializable(meta_data[category]))
                        examples_this_iteration += 1  # Increment when example is valid
                    else:
                        # Store rejected example
                        if category == 'gt1':
                            rejected_data[category][new_key] = {'gt1': desired_label}
                        else:
                            rejected_data[category][new_key] = {
                                category: desired_label,
                                'gt1': meta_info['gt1']
                            }
                        update_json(rejected_output_dirs[category], convert_to_serializable(rejected_data[category]))
                        examples_this_iteration += 1  # Increment even for rejected examples
                        
        # Print timing statistics only if we processed any examples
        if examples_this_iteration > 0:
            example_time = time.time() - example_start_time
            example_count += examples_this_iteration
            # avg_time_per_example = example_time / examples_this_iteration
            print(f"Time this iteration: {example_time:.2f}s. ")
            
        # print the counter for each category
        print('---'*10)
        for counter_ in counters:
            print(f"{counter_}: {counters[counter_]}")

        # Check if all categories reached the limit
        done = all(all(count >= max_count for count, max_count in zip(count_list, desired_per_label_count)) for count_list in counters.values())
        if done:
            print("Reached the limit for all categories. Exiting.")
            break

    print("Finished processing.")

if __name__ == '__main__':
    main()
