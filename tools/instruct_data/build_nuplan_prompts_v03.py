import os
import json
from tqdm import tqdm
import time
import glob
import numpy as np


def get_distance_and_angle(location_str):
    current_location = np.array([float(i) for i in ''.join(''.join(location_str.split('(')).split(')')).split(', ')])
    current_distance = np.linalg.norm(current_location).round(2)
    current_angle = int(np.degrees(np.arctan2(current_location[1], current_location[0])))
    return current_distance, current_angle


def parse_json_file(json_path):
    # Parse the jason file that represents only one sample.
    with open(json_path, 'r') as file:
        sample_dict = json.load(file)
    return sample_dict

def get_acts(templates, direction_classes):
    ## loading and filtering ground truth and contrastive acts
    gt_templates, contrastive_templates, positive_templates = [], [], []
    gt_act = templates[0]['Direction_cls']
    contrastive_acts = []
    positive_acts_not_gt = []
    for template_i in templates:
        if template_i['Label']=='gt':
            gt_templates.append(template_i)
        if template_i['Label']=='negative':
            contrastive_acts.append(template_i['Direction_cls'])
            contrastive_templates.append(template_i)
        if template_i['Label']=='possible direction not gt':
            positive_acts_not_gt.append(template_i['Direction_cls'])
            positive_templates.append(template_i)
    if 'straight' in direction_classes[gt_act] or True in ['straight' in direction_i for direction_i in direction_classes[positive_acts_not_gt]]:
        contrastive_templates = [contrastive_templates[i] for i in range(len(contrastive_acts)) if 'straight' not in direction_classes[contrastive_acts[i]]]
        contrastive_acts = [act_i for act_i in contrastive_acts if 'straight' not in direction_classes[act_i]]
    
    return gt_act, contrastive_acts, positive_acts_not_gt, gt_templates, contrastive_templates, positive_templates

def get_traffic_prompt(map):
    # traffic_prompt = "List of stop signs and traffic lights current states if they exist (can be used for reasoning of all tasks, can be combined with the Instruction of Task2 or Task3 or Task4)\n"
    no_traffic_light = True
    no_stop_sign = True
    if len(map)==0:
        traffic_prompt = "There is no close traffic light."
        stop_prompt = "There is no close stop sign."
        return traffic_prompt, stop_prompt
    
    for map_k in map:
        if 'Stop' in map_k:
            no_stop_sign = False
        if 'Traffic' in map_k:
            no_traffic_light = False
    if no_stop_sign and no_traffic_light:
        traffic_prompt = "There is no close traffic light."
        stop_prompt = "There is no close stop sign."
        return traffic_prompt, stop_prompt

    if no_stop_sign:
        stop_prompt = "There is no close stop sign."
    if no_traffic_light:
        traffic_prompt = "There is no close traffic light."
    
    stop_signs_dist = []
    stop_signs_angle = []
    traffic_lights_dist = []
    traffic_lights_angle = []
    traffic_lights_state = []
    for map_k in map:
        if 'location' in map_k:
            distance, angle = get_distance_and_angle(map[map_k])
            if distance>200:
                continue
            if abs(angle)<120:
                if abs(angle)<=25:
                    heading='in front'
                elif angle>0:
                    heading='on the left'
                else:
                    heading='on the right'
            else:
                heading='behind'
                
            if 'Stop' in map_k:
                stop_signs_dist.append(distance)
                stop_signs_angle.append(heading)
            else:
                traffic_lights_dist.append(distance)
                traffic_lights_angle.append(heading)
                traffic_lights_state.append(map[map_k.replace('location', 'state')])
    
    if len(traffic_lights_dist)>0:
        if len(traffic_lights_dist)==1:
            traffic_prompt = f"There is one {traffic_lights_state[0]} traffic light {traffic_lights_angle[0]}, {traffic_lights_dist[0]} m away."
        else:
            traffic_prompt = f"There are {len(traffic_lights_dist)} close traffic lights - " +', '.join([f"one {traffic_lights_state[i]} {traffic_lights_angle[i]} {traffic_lights_dist[i]} m away" for i in range(len(traffic_lights_state))])+'.'
    else:
        traffic_prompt = "There is no close traffic light."
    if len(stop_signs_dist)>0:
        if len(stop_signs_dist)==1:
            stop_prompt = f"There is one stop sign {stop_signs_angle[0]}, {stop_signs_dist[0]} m away."
        else:
            stop_prompt = f"There are {len(stop_signs_dist)} close stop signs - " +', '.join([f"one {traffic_lights_angle[i]} {traffic_lights_dist[i]} m away" for i in range(len(traffic_lights_state))])+'.'
    else:
        stop_prompt = "There is no close stop sign."
    
    return traffic_prompt, stop_prompt

def generate_meta_caption(agent ,map):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    # """
    # ['stationary', 'constant velocity', 'mild acceleration', 'mild deceleration', 'moderate acceleration', 'moderate deceleration', 'aggressive acceleration', 'aggressive deceleration', 'extreme acceleration', 'extreme deceleration']
    direction = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to8'] else agent['Agent-1']['direction 0.1to8']
    direction1 = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to4'] else agent['Agent-1']['direction 0.1to4']
    direction2 = "stay stationary" if "stationary" in agent['Agent-1']['direction 4to8'] else agent['Agent-1']['direction 4to8']
    accel1 = f"stay {agent['Agent-1']['acceleration 0.1to4']}" if 'stationary' == agent['Agent-1']['acceleration 0.1to4'] else f"have {agent['Agent-1']['acceleration 0.1to4']}"
    accel2 = f"stay {agent['Agent-1']['acceleration 4to8']}" if 'stationary' == agent['Agent-1']['acceleration 4to8'] else f"have {agent['Agent-1']['acceleration 4to8']}"
    speed0, speed1, speed2 = agent['Agent-1']['Current Speed'], agent['Agent-1']['speed 4'], agent['Agent-1']['speed 8']
    
    meta_caption = f"""Assume the origin is the current ego vehicle location and it is initially heading East.\n"""
    meta_caption += f"""- [Direction]: It will {direction} by first {direction1} in 4 seconds, then {direction2}.\n"""
    meta_caption += f"""- [Speed and acceleration]: The current ego vehicle's speed is {speed0}. It will first {accel1} reaching a speed of {speed1} in 4 seconds, then will {accel2} reaching speed {speed2}.\n"""
    meta_caption += f"""- [Other agents]: """
    
    num_agents = 5
    for agent_i_key in agent.keys():
        if agent_i_key != 'Agent-1':
            num_agents-=1
            if num_agents<0:
                break
            agent_i = agent[agent_i_key]
            agent_type = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
            current_distance,current_angle = get_distance_and_angle(agent_i['Current Location'])
            if abs(current_angle)<25:
                relative_location = 'in front'
            elif current_angle<120 and current_angle>25:
                relative_location = 'on the left'
            elif current_angle>-120 and current_angle<-25:
                relative_location = 'on the right'
            else:
                relative_location = 'behind'
            
            heading = agent_i['Current Heading']
            heading = int(heading.replace(' degrees', ''))
            if abs(heading)<22.5:
                heading='East'
            elif abs(heading)<67.5:
                heading = 'North-East' if heading>0 else "South-East"
            elif abs(heading)<112.5:
                heading = 'North' if heading>0 else "South"
            elif abs(heading)<157.5:
                heading = 'North-West' if heading>0 else "South-West"
            elif abs(heading)<181:
                heading = 'West'
            else:
                heading = 'Unknown'
            current_speed = agent_i['Current Speed']
            # if num_agents>0:
            meta_caption+= f"""{agent_i_key} is a {agent_type[agent_i['Type']]}, it is {current_distance} m {relative_location} of the ego vehicle, it's current speed is {current_speed}, and is heading {heading}. """

    
    traffic_prompt, stop_prompt = get_traffic_prompt(map)
    meta_caption += f"""\n- [Traffic light]: {traffic_prompt}\n"""
    meta_caption += f"""- [Stop sign]: {stop_prompt}\n"""
    
    # print(meta_caption)
    # prompt = prompt + get_traffic_prompt(map)

    return meta_caption

if __name__ =="__main__":
    print("...")
    
    data_dir = "json_example"
    agent_dir = data_dir+'/agents.json'
    map_dir = data_dir+'/map.json'
    # Set IMOTION_SCENE_JSON_DIR to the directory of per-scene agent/map JSON
    # produced by the GameFormer preprocessing step.
    base_dir = os.environ.get(
        'IMOTION_SCENE_JSON_DIR',
        'data/processed/waymo/gameformer/train_agentJsons',
    )
    files = glob.glob(base_dir+'/*')
    direction_classes=np.array(['stationary', 'move straight', 'move straight veering right', 'move straight veering left', 'turn right', 'turn left', 'take right U-turn', 'take left U-turn'])
    json_out = {}
    max_num_files = 50000
    for file in tqdm(files):
        filename = file.split('/')[-1].replace('.npz', '')
        agent_dir = '/'.join(file.split('/')[:-1])+'_agentJsons/' + file.split('/')[-1].replace('npz', 'json')
        map_dir = '/'.join(file.split('/')[:-1])+'_mapJsons/' + file.split('/')[-1].replace('npz', 'json')
        template_llm_dir = '/'.join(file.split('/')[:-1])+'_templateLLM/' + file.split('/')[-1].replace('npz', 'txt')
        agent_json = parse_json_file(json_path=agent_dir)
        map_json = parse_json_file(json_path=map_dir)

        gpt_prompt = generate_meta_caption(agent_json, map_json)

        with open(template_llm_dir) as f:
            lines = f.readlines()
        templates = [json.loads(line) for line in lines]
        # contrastive_acts, gt_templates, contrastive_templates = self.get_contrastive_act(templates)
        gt_act, contrastive_acts, positive_acts, gt_templates, contrastive_templates, positive_templates = get_acts(templates, direction_classes)
        gpt_prompt += f"""- [No lanes directions]: There are no lanes that allow the directions """+'('+', '.join(list(direction_classes[contrastive_acts]))+').'


        json_out[filename] = gpt_prompt
        max_num_files-=1
        if max_num_files==0:
            break
    
    print('')

    # Save the dictionary to a JSON file
    with open('gpt_ready_content.json', 'w') as json_file:
        json.dump(json_out, json_file, indent=4)

        