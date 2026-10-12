import os
import json
from tqdm import tqdm
import time
import glob
import numpy as np

def get_input_prompt():
    return """I am guiding someone to drive (referred to as the ego driver). Driving is on the left side of the road. A red traffic light indicates stopping for controlled lanes, and a yellow light warns that the signal will turn green or red. We observe objects like vehicles, pedestrians, cyclists, traffic lights, and stop signs ahead.

Each vehicle, including the ego driver's, has a heading direction measured in degrees. 0 degrees indicates heading east, 90 degrees north, -90 degrees south, and 180 or -180 degrees west. Based on (x, y) location coordinates, we calculate each vehicle's relative Euclidean distance and angle to the ego driver. Since the ego is moving east, negative angles mean the object is to the right, positive angles to the left. If an angle is greater than 90 degrees in magnitude, it's a sharp angle behind (e.g., 134 degrees is behind-left, -150 is behind-right). Vehicles in the same lane should have similar heading angles. If a vehicle is driving in the opposite direction and to the left of the ego driver, it's in the opposite lane.

Please follow these guidelines:
- The driver can accept the guidance and reason how to execute it based on surrounding objects, rules, and common sense. Alternatively, they can accept it cautiously or reject it if it's a high risk, unreasonable, or harmful.
- The ego driver should be extremely cautious of agents in extreme risk zones and proceed cautiously in high-risk zones. If agents are outside these zones, caution isn't needed.
- Safety zones assume the vehicle is not stationary and are based on the ego driver's maximum observed speed in the last second. If stationary, a 10m range is a high-risk zone and 3m is an extreme risk zone.
- Include instructions for direction, speed, speed category, and relative location to other agents.
- Direction categories: stationary, straight, veering left, veering right, left, right, U-turn left, U-turn right.
- Speed categories: stationary, very slow, slow, moderate, fast, very fast, extremely fast.
- Relative location is semantic, e.g., (close to, far from, behind, in front of, next to).
- Use provided future direction and speed (0 to 8 seconds ahead, 0 to 4 seconds ahead, and 4 to 8 seconds ahead) for designing instructions.
- Reasoning should consider the current and past states of surrounding agents and how following the instruction fits with that information. Future acceleration and movement can help to reason about the ego driver's actions.
- Movement or acceleration implies speed and location values, e.g., stopping means 0 km/h or close.
- Correct provided information if needed; if speed or movement is unknown and the vehicle hasn't moved, it's stationary.
- The ego driver is heading 0 degrees (east).
- Pay attention to objects directly ahead when moving straight and those on the sides when turning. And to objects behind within proximity when stopping or slowing down.
- Be cautious of interactions with far objects or those in other lanes that could come in front of or next to the ego driver based on speed and movement.
- Monitor pedestrians within proximity who might cross at a red traffic light.
- Assess the future behavior of the ego driver to classify as "<Accepted>" or "<Accepted_with_caution>" or "<Reject>" based on common sense and surroundings. Even if high risk, label it "<Accepted_with_caution>" if fitting.

Reasoning should be based on:
- The ego driver's future, current, or past state lists.
- Lists of other agents' states, traffic light, and stop sign information.
- Relative location and direction to each agent.
- Expected future location based on current speed and location.
- Varying attention levels depending on hazard potential (speed, direction changes, stops).
- Greater attention to agents, traffic lights, and stop signs in front and nearby.
- How instructions will be executed safely, cautiously, or rejected, considering location, descriptions, surroundings, and future state of the ego driver.
- Use all provided information for reasoning, but don't invent details not supplied (other than straightforward ones, like relative distance and relative angle of other objects).

Now design 9 Instruction-Reasoning-Decision of instructions and driver reasoning responses and decide whether to accept the instruction, accept it with caution, or reject it while reasoning about the feasibility or risks.

Format each Instruction-Reasoning-Decision in a single line as a JSON dictionary. The JSON should have the following format: {{"Instruction": "xxx", "Reasoning": "xxx", "Decision": "<Accepted>/<Rejected>/<Accepted_with_caution>/<Out_of_domain>"}}. Only output 9 lines of single-line JSON. Do not include any other explanation.

The tasks to generate:
- Task1: Include 2 Instruction-Reasoning-Decision pairs that can be executed with either an "<Accepted>" or "<Accepted_with_caution>" Decision label based on your reasoning and understanding of the surroundings. The instruction should be based on variations or combinations of the provided ego driver future direction and speed information (List of ego driver future direction and speed information), not necessarily including all of them but based on them. The reasoning should be based on the rules provided earlier and your reasoning using all the provided information.
- Task2: Include 2 Instruction-Reasoning-Decision pairs that can be executed with either an "<Accepted>" or "<Accepted_with_caution>" Decision label based on your reasoning and understanding of the surroundings. The instruction should include relative current and future locations of other agents in a semantic way (e.g., moving far from, close to, to the right/left, or behind), with future direction and speed information. Still, it should be based on the provided future behavior of the ego driver only (List of ego driver future direction and speed information, List of ego driver future locations) and the current state of the surroundings. The reasoning should be based on the rules provided earlier and your reasoning using all the provided information.
- Task3: Include 2 Instruction-Reasoning-Decision pairs that are not feasible due to a reckless, unsafe, or extremely high-risk driving Decision. These instructions should contradict the provided future driving behavior of the ego driver. For example, accelerating while the vehicle is moving fast or requesting the driver to stop in the middle of the road while other vehicles are moving behind the ego driver. Include diverse reckless instructions with reasoning to reject them. These should have a Decision label of "<Rejected>."
- Task4: Include one Instruction-Reasoning-Decision with a random instruction that is not about driving. The reasoning should be, "As an AI Driver, what you provided is out of my scope," and the Decision should be "<Out_of_domain>."

Example outputs of each task:
- Task1: {{"Instruction": "Turn right, by first moving straight, then turn right at a slow speed", "Reasoning": "I can safely take a right turn first, decelerating on the road, then turning right at a slow speed because the traffic light is green, and no vehicles are in front of me or within high-risk proximity.", "Decision": "<Accepted>"}}
- Task2: {{"Instruction": "Move straight while increasing the speed to 50 km/h", "Reasoning": "I can cautiously start to move and then aggressively accelerate with caution. I will first accelerate gently because there is a car in the front-right of us driving at a slow speed. Once we pass the car, I will accelerate extremely. We need to be cautious of the vehicle on the right that is traveling at a slower speed when accelerating.", "Decision": "<Accepted_with_caution>"}}
- Task2: {{"Instruction": "Move straight, closer to the vehicle in front", "Reasoning": "I can move straight while gently accelerating, but we need to be cautious as we are moving fast and the vehicle directly in front of us is in a high-risk zone, so we must decelerate if it stops.", "Decision": "<Accepted_with_caution>"}}
- Task3: {{"Instruction": "Slow down to a speed of 2 km/h", "Reasoning": "I cannot decelerate this aggressively due to vehicles behind me moving at high speed.", "Decision": "<Rejected>"}}
- Task3: {{"Instruction": "Move straight fast", "Reasoning": "I cannot move straight, there is a red traffic light just in front of us.", "Decision": "<Rejected>"}}
- Task3: {{"Instruction": "Move straight at a very fast speed", "Reasoning": "I cannot drive very fast; there are a lot of vehicles in proximity and traveling at low speeds. Traveling very fast will be very high risk.", "Decision": "<Rejected>"}}
- Task4: {{"Instruction": "What time of day is it?", "Reasoning": "As an AI Driver, what you provided is out of my scope.", "Decision": "<Out_of_domain>"}}
    
"""

def get_ego_prompt(ego_agent):
    return f"""List of current and past ego vehicle states (can be used for reasoning of all tasks)
- Current location: {ego_agent['Current Location']}
- Current heading: {ego_agent['Current Heading']}
- Current speed: {ego_agent['Current Speed']}

List of risk zones based on maximum observed speed of the ego vehicle (can be used for reasoning of all tasks)
- Extreme high risk zone range: {ego_agent['Distance in 1s']}
- High risk zone range: {ego_agent['Distance in 3s']}
- Safe zone range but within proximity: {ego_agent['Distance in 8s']}
If extreme risk zone is less than 3m, consider it as 3m. If high risk zone is less than 10m, consider it as 10m. If safe zone is less than 20m, consider it as 20m.

List of ego vehicle future direction and speed information (The only information used for the Instruction of Task1. Can be used for the Instruction of Task2. Should not be used for Task3)
- Direction from 0 to 8 seconds: {ego_agent['direction 0.1to8']}
- Speed category from 0 to 8 seconds: {ego_agent['speed 0.1to8']}
- Direction from 0 to 4 seconds: {ego_agent['direction 0.1to4']}
- Speed category  from 0 to 4 seconds: {ego_agent['speed 0.1to4']}
- Speed at 4 seconds: {ego_agent['speed 4']}
- Direction from 4 to 8 seconds: {ego_agent['direction 4to8']}
- Speed category  from 4 to 8 seconds: {ego_agent['speed 4to8']}
- Speed at 8 seconds: {ego_agent['speed 8']}

List of ego vehicle future locations (can be used for Instruction of Task2, the Reasoning only of Task1, should not be used for Task3)
- Location after 4 seconds: {ego_agent['Future Location in 4s']}
- Heading after 4 seconds: {ego_agent['Future Heading in 4s']}
- Location after 4 seconds: {ego_agent['Future Location in 4s']}
- Heading after 8 seconds: {ego_agent['Future Heading in 8s']}

List of ego vehicle future acceleration and movement (no acceleration or movement information used for the Instruction of Task1 and Task2, can be used for Reasoning only of Task1 and Task2, should not be used for Task3)
- Acceleration category from 0 to 8 seconds: {ego_agent['acceleration 0.1to8']}
- Movement category from 0 to 8 seconds: {ego_agent['movement 0.1to8']}
- Acceleration category from 0 to 4 seconds: {ego_agent['acceleration 0.1to4']}
- Movement category from 0 to 4 seconds: {ego_agent['movement 0.1to4']}
- Acceleration category from 4 to 8 seconds: {ego_agent['acceleration 4to8']}
- Movement category from 4 to 8 seconds: {ego_agent['movement 4to8']}

"""

def get_distance_and_angle(location_str):
    current_location = np.array([float(i) for i in ''.join(''.join(location_str.split('(')).split(')')).split(', ')])
    current_distance = np.linalg.norm(current_location).round(2)
    current_angle = int(np.degrees(np.arctan2(current_location[1], current_location[0])))
    return current_distance, current_angle

def get_traffic_prompt(map):
    traffic_prompt = "List of stop signs and traffic lights current states if they exist (can be used for reasoning and supporting the design of the instructions)\n"
    no_traffic_light = True
    no_stop_sign = True
    if len(map)==0:
        traffic_prompt+= "- There are no stop signs or traffic lights.\n"
        return traffic_prompt
    
    for map_k in map:
        if 'Stop' in map_k:
            no_stop_sign = False
        if 'Traffic' in map_k:
            no_traffic_light = False
    if no_stop_sign and no_traffic_light:
        traffic_prompt+= "- There are no stop signs or traffic lights.\n"
        return traffic_prompt

    if no_stop_sign:
        traffic_prompt+= "- There are no stop signs.\n"
    if no_traffic_light:
        traffic_prompt+= "- There are no traffic lights.\n"
    
    for map_k in map:
        if 'location' in map_k:
            distance, angle = get_distance_and_angle(map[map_k])
            traffic_prompt+= f"- {map_k.replace('location', 'distance from ego')}: {distance} m\n"
            traffic_prompt+= f"- {map_k.replace('location', 'angle from ego')}: {angle} degrees\n"
        else:
            traffic_prompt+= f"- {map_k}: {map[map_k]}\n"
    
    return traffic_prompt

def get_agent_prompt(agent_i, i):
    agent_type = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']
    current_distance,current_angle = get_distance_and_angle(agent_i['Current Location'])
    past_distance,past_angle = get_distance_and_angle(agent_i['Location before 1s'])
    return f"""List of current and past {i} states (can be used for reasoning and supporting the design of the instructions Task2)
- Agent type: {agent_type[agent_i['Type']]}
- Distance from ego before 1 second:  {past_distance} m
- Angle from ego before 1 second:  {past_angle} degrees
- Current distance from ego:  {current_distance} m
- Current angle from ego:  {current_angle} degrees
- Heading before 1 second: {agent_i['Heading before 1s']}
- Current heading: {agent_i['Current Heading']}
- Speed before 1 second: {agent_i['Speed before 1s']}
- Current speed: {agent_i['Current Speed']}

"""
# return f"""List of current and past {i} states (can be used for reasoning and supporting the design of the instructions Task2)
# - Agent type: {agent_type[agent_i['Type']]}
# - Location before 1 second: {agent_i['Location before 1s']}
# - Distance to ego before 1 second:  {past_distance} m
# - Angle to ego before 1 second:  {past_angle} degrees from ego
# - Current location: {agent_i['Current Location']}
# - Current distance to ego:  {current_distance} m
# - Current angle to ego:  {current_angle} degrees from ego
# - Heading before 1 second: {agent_i['Heading before 1s']}
# - Current heading: {agent_i['Current Heading']}
# - Speed before 1 second: {agent_i['Speed before 1s']}
# - Current speed: {agent_i['Current Speed']}

# """

def list_files_in_dir(in_path):
    # list files in dir
    files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    return files

def parse_json_file(json_path):
    # Parse the jason file that represents only one sample.
    with open(json_path, 'r') as file:
        sample_dict = json.load(file)
    return sample_dict

def generate_meta_caption(agent ,map):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    # """
    
    prompt = get_input_prompt()
    # get_ego_prompt(agent['Agent-1'])
    prompt = prompt + get_ego_prompt(agent['Agent-1'])
    for agent_i in agent:
        if agent_i != 'Agent-1':
            prompt = prompt + get_agent_prompt(agent[agent_i], agent_i)

    prompt = prompt + get_traffic_prompt(map)

    return prompt
    
    # return template, instruct_template1, instruct_template2, instruct_template3, instruct_template4


if __name__ =="__main__":
    print("...")
    
    data_dir = "json_example"
    agent_dir = data_dir+'/agents.json'
    map_dir = data_dir+'/map.json'
    agent_json = parse_json_file(json_path=agent_dir)
    map_json = parse_json_file(json_path=map_dir)
    # for k,v in agent_json.items():
    #     print("---"*10)
    #     if 'Agent-1' == k:
    #         for k_,v_ in v.items():
    #             print(f"{k_}: {v_}")
    #     else:
    #         print("---"*10)
    #         print(k)
    #         for k_,v_ in v.items():
    #             print(f"{k_}: {v_}")
    # print("***"*10)
    # for k,v in map_json.items():
    #     print(f"{k}: {v}") 
    
    gpt_prompt = generate_meta_caption(agent_json, map_json)
    with open('example.txt', 'w') as file: 
      file.writelines(gpt_prompt)


    # data_dir = "data/processed/waymo/gameformer/small_validation_5may_fullmap_val"
    # # data_dir = "data/raw/waymo/training_interactive_p_29feb_json"
    # samples_files_names = glob.glob(data_dir+'/*')
    # # samples_files_names = list_files_in_dir(in_path=data_dir)
    # print("......")
    # for sample_file_name in tqdm(samples_files_names):
    #     sample_name = sample_file_name.split('.')[0]
    #     sample_file_name.split('.')[0].split('/')[-1]
    #     agent_dir = data_dir+'_agentJsons/agent_' + sample_file_name.split('.')[0].split('/')[-1]+'.json'
    #     map_dir = data_dir+'_mapJsons/map_' + sample_file_name.split('.')[0].split('/')[-1]+'.json'
    #     # 1- Parse the json file:
    #     agent_json = parse_json_file(json_path=agent_dir)
    #     map_json = parse_json_file(json_path=map_dir)
        

    #     for k,v in agent_json.items():
    #         print("---"*10)
    #         if 'Agent-1' == k:
    #             for k_,v_ in v.items():
    #                 print(f"{k_}: {v_}")
    #         else:
    #             print("---"*10)
    #             print(k)
    #             for k_,v_ in v.items():
    #                 print(f"{k_}: {v_}")

            
    #     for k,v in map_json.items():
    #         print(f"{k}: {v}")
    #     generate_meta_caption(agent_json, map_json)


        