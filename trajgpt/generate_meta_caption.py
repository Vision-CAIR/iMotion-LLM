import os
import json
from tqdm import tqdm
import time
import glob

# in_path = "<internal_waymo_dataset_root>/validation_interactive_p_29feb_00_json"
# in_path = "<internal_waymo_dataset_root>/validation_interactive_p_29feb_00"
# files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
def list_files_in_dir(in_path):
    # list files in dir
    files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    
    return files

def parse_json_file(json_path):
    # Parse the jason file that represents only one sample.
    with open(json_path, 'r') as file:
        sample_dict = json.load(file)

    return sample_dict


def get_vehicle_names_for_azone(list_of_vehicles):
    vehicle_names_for_azone = ""
    if len(list_of_vehicles):
        vehicle_names_for_azone = " namely, "
        for i, vehicle_name in enumerate(list_of_vehicles):
            if i== len(list_of_vehicles)-1:
                vehicle_names_for_azone = vehicle_names_for_azone + vehicle_name+"."
            else:
                vehicle_names_for_azone = vehicle_names_for_azone + vehicle_name+", "
            
    return vehicle_names_for_azone


def get_stop_sign_info(sample_dict):
    stop_sign_str = "There is no stop sign ahead."
    if "Agent-A stop sign state" in sample_dict.keys():
        stop_sign_str = f"""There is a stop sign \
{sample_dict['Agent-A stop sign relative location'].lower()} \
{sample_dict['Agent-A stop sign distance']} meters away."""
    
    return stop_sign_str


def get_traffic_sign_info(sample_dict):
    traffic_sign_str = "There is no traffic light ahead."
    if "Agent-A traffic sign state" in sample_dict.keys():
        traffic_sign_str = f"""There is a {sample_dict['Agent-A traffic sign state']} traffic sign \
{sample_dict['Agent-A traffic sign relative location'].lower()} \
{sample_dict['Agent-A traffic sign distance']} meters away."""
    
    return traffic_sign_str

def get_interactive_info(sample_dict):
    info = ""
    if 'Agent-A view' in sample_dict.keys():
        info = f"6- {sample_dict['Agent-A view']}"
    return info

def generate_meta_caption(sample_dict):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    """
    vehicles_names_in_zone1 = get_vehicle_names_for_azone(list_of_vehicles=sample_dict["Agent-A 1sec agents"])
    vehicles_names_in_zone2 = get_vehicle_names_for_azone(list_of_vehicles=sample_dict["Agent-A 3sec agents"])
    vehicles_names_in_zone3 = get_vehicle_names_for_azone(list_of_vehicles=sample_dict["Agent-A 8sec agents"])
    # for k in list(sample_dict.keys()):
    #     print(k)
    # point_numbering = [1,2,3,4,5,6,7]
#     template = f"""The ego vehicle, termed Agent-A, can safely {sample_dict['Agent-A turn']} while {sample_dict['Agent-A move']}, 
# by first {sample_dict['Agent-A turn_1']} while {sample_dict['Agent-A move_1']} then {sample_dict['Agent-A turn_2']} while {sample_dict['Agent-A move_2']}.
# Because, the current scene information as follows:
# 1- Agent-A vehicle's speed is {int(float(sample_dict['Agent-A speed'])*60*60/1000)} km/h and it is currently {sample_dict['Agent-A current map state']} and will {sample_dict['Agent-A future map state']}{f", sample_dict{sample_dict['Agent-A caution']}" if sample_dict['Agent-A caution']!='' else ''}.
# 2- The close and risky zone based on speed, which covers the radius of {sample_dict['Agent-A 1sec distance']} meters, contains {len(sample_dict['Agent-A 1sec agents'])} agents{vehicles_names_in_zone1}
# 3- The medium zone, based speed and the 3 second rule in safe driving, which covers the radius of {sample_dict['Agent-A 3sec distance']} meters, contains {len(sample_dict['Agent-A 3sec agents'])} agents{vehicles_names_in_zone2}
# 4- The far and safe zone based on speed, which covers the radius of {sample_dict['Agent-A 8sec distance']} meters, contains {len(sample_dict['Agent-A 8sec agents'])} vehicles{vehicles_names_in_zone3}
# 5- {get_stop_sign_info(sample_dict)}
# 6- {get_traffic_sign_info(sample_dict)}
# {get_interactive_info(sample_dict)}"""

    template = f"""The ego vehicle, termed Agent-A, can safely {sample_dict['Agent-A turn']} while {sample_dict['Agent-A move']}, 
by first {sample_dict['Agent-A turn_1']} while {sample_dict['Agent-A move_1']} then {sample_dict['Agent-A turn_2']} while {sample_dict['Agent-A move_2']}.
Because, the current scene information as follows:
1- Agent-A vehicle's speed is {int(float(sample_dict['Agent-A speed'])*60*60/1000)} km/h and it is currently {sample_dict['Agent-A current map state']} and will {sample_dict['Agent-A future map state']}{f", sample_dict{sample_dict['Agent-A caution']}" if sample_dict['Agent-A caution']!='' else ''}.
2- The close and risky zone based on speed, which covers the radius of {sample_dict['Agent-A 1sec distance']} meters, contains {len(sample_dict['Agent-A 1sec agents'])} agents{vehicles_names_in_zone1}
3- The medium and caution zone, based on speed and the 3 second rule in safe driving, which covers the radius of {sample_dict['Agent-A 3sec distance']} meters, contains {len(sample_dict['Agent-A 3sec agents'])} agents{vehicles_names_in_zone2}
4- {get_stop_sign_info(sample_dict)}
5- {get_traffic_sign_info(sample_dict)}
{get_interactive_info(sample_dict)}"""

    instruct_template1 = f"""Make the ego vehicle, termed Agent-A, {sample_dict['Agent-A turn']}."""
    instruct_template2 = f"""Make the ego vehicle, termed Agent-A, {sample_dict['Agent-A move']}."""
    instruct_template3 = f"""Make the ego vehicle, termed Agent-A, {sample_dict['Agent-A turn']} while {sample_dict['Agent-A move']}."""
    instruct_template4 = f"""Make the ego vehicle, termed Agent-A, {sample_dict['Agent-A turn_1']} while {sample_dict['Agent-A move_1']} then {sample_dict['Agent-A turn_2']} while {sample_dict['Agent-A move_2']}."""
    
    return template, instruct_template1, instruct_template2, instruct_template3, instruct_template4


if __name__ =="__main__":
    print("...")
    data_dir = "<internal_waymo_dataset_root>/validation_interactive_p_29feb_json"
    # data_dir = "<internal_waymo_dataset_root>/training_interactive_p_29feb_json"
    samples_files_names = glob.glob(data_dir+'/*')
    # samples_files_names = list_files_in_dir(in_path=data_dir)
    print("......")
    for sample_file_name in tqdm(samples_files_names):
        sample_name = sample_file_name.split('.')[0]
        # 1- Parse the json file:
        sample_dict = parse_json_file(json_path=os.path.join(data_dir, sample_file_name))
        # print(sample_dict)
        # print("-"*50)cd /
        # print("-"*50)
        # print("-"*50)
        generate_meta_caption(sample_dict)
        # print(generate_meta_caption(sample_dict))
        # time.sleep(5)
        # break