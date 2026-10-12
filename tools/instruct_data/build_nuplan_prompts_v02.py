import os
import json
from tqdm import tqdm
import time
import glob
import numpy as np

def get_context():
    return """You are a helpful assistant that can reason about driving scenarios, and you only reply with JSON.

I am guiding someone to drive (referred to as the ego driver). Driving is on the left side of the road. We observe objects like vehicles, pedestrians, cyclists, traffic lights, and stop signs ahead.

Each vehicle, including the ego driver's, has a heading direction measured in degrees. 0 degrees indicates heading east (straight), 90 degrees north (left), -90 degrees south (right), and 180 or -180 degrees west (back). Each vehicle has a relative Euclidean distance and angle to the ego driver. Since the ego is moving east, relative positive angles from the ego are to the left, and negative angles mean the object is to the right. If an angle is greater than 90 degrees in magnitude, it's a sharp angle behind (e.g., 134 degrees is behind-left, -150 is behind-right). Vehicles in the same lane should have similar heading angles. If a vehicle is driving in the opposite direction and to the left of the ego driver, it's in the opposite lane. If it has around 90 degrees in magnitude heading angle, it is moving cross-wise.

Please follow these guidelines:
- Instruction includes the direction of travel, and the speed category of travel.
- Reason include executing the direction of travel, and how it is executed in two steps, similarly for speed. Then reason why it is safe to do so based on surrounding objects distances and relative locations and states.
- The ego driver should be extremely cautious of agents in extreme risk zones and proceed cautiously in high-risk zones. If agents are outside these zones, caution isn't needed.
- Safety zones assume the vehicle is not stationary and are based on the ego driver's maximum observed speed in the last second. If stationary, a 10m range is a high-risk zone and 3m is an extreme risk zone.
- Include instructions for direction, speed, speed category, and relative location to other agents.
- Direction categories: stationary (no motion), straight, veering left, veering right, left, right, U-turn left, U-turn right.
- Speed categories: stationary (no motion), very slow, slow, moderate, fast, very fast, extremely fast.
- Reasoning should consider the current and past states of surrounding agents and how following the instruction fits with that information. 
- Future information can only be used with "<Accepted>" Instructions, to narrate how the action will be executed step by step in terms of direction, speed, movement, and acceleration.
- Movement or acceleration implies speed and location values, e.g., stopping means 0 km/h or close.
- Correct provided information if needed; if speed or movement is unknown and the vehicle hasn't moved, it's stationary.
- Pay attention to objects directly ahead when moving straight and those on the sides when turning. And to objects behind within proximity when stopping or slowing down.
- Be cautious of interactions with far objects or those in other lanes only if they could come in front of or next to the ego driver based on their relative location, speed and movement.
- Monitor pedestrians within proximity who might cross at a red traffic light.
- The provided future direction of the ego vehicle should always be included in the design of "<Accepted>" Instructions.
- Always adhere to the provided future direction (the overall direction) for the ego driver when designing Instructions. Use it to narrate the execution of the Instruction in the Reason, but not as a reason of safety or feasibility. 
- Do not create new Instructions that contradict the provided future information.
- Do not use the provided future information reason, for example do not say things like "because it contradicts the provided future information".
- Base the assessment of traffic lights and stop signs and other agents strictly on their stated relative locations and angles. Avoid making assumptions about their presence or state if not directly provided.
- Rephrase the provided information if needed in the design of Instruction or Reasoning, as they can be written in a template way and not in a natural way of speaking or writing.
- Reasoning should include information about other objects relative locations.
- In reasoning pay greater attention to agents, traffic lights, and stop signs in front and nearby.
- In reasoning explain how and why instructions will be executed safely, cautiously, or rejected.
- Don't invent details not supplied. Even when designing an Instruction that will be labeled as "<Rejected>".
- Interactive agent of interest is expected to interact with the ego agent in the future where they could drive close to each other or pass by each other. Try to include it in the reasoning.
- Agents and objects in front of us (with an angle from ego of 0) and within proximity should have the highest level of attention.
- Positive relative angle from ego indicate agent is on the left, negative indicate on the right, close to 180 or -180 indicate behind.
- You should include Agent-2 in the reasoning, we know that it is essential to expect interaction with it.


Now design 7 Instruction-Reasoning-Decision based on the above guidelines.

Format each Instruction-Reasoning-Decision in a single line as a JSON dictionary. The JSON should have the following format: {{"Instruction": "xxx", "Reasoning": "xxx", "Decision": "<Accepted>/<Rejected>/<Out_of_domain>"}}. 
Only output 7 lines of single-line JSON. Do not include any other explanation.

The tasks to generate:
- Task1: You must include 2 examples of Instruction-Reasoning-Decision pairs that can be executed with "<Accepted>" Decision label. The instruction should be based on the provided ego driver future direction and speed information, It should always include the overall future direction, and could include the per step information. The reasoning should start by explaining how the driver will execute the instruction, then reason why based on the guidelines provided earlier, reasoning and common sense.
- Task2: You must include 2 examples of Instruction-Reasoning-Decision pairs that can be executed with "<Accepted>" Decision label. The instruction additional to Task1, should include relative current and future locations of other agents in a semantic way (e.g., moving far from, close to, to the right/left, behind, past something) while still including direction and speed information provided. This still should match the overall direction and speed provided. The reasoning should start by explaining how the driver will execute the instruction, then reason why based on the guidelines provided earlier, reasoning and common sense.
- Task3: You must include 1 example of Instruction-Reasoning-Decision pairs that are not feasible due to a reckless, unsafe, or extremely high-risk driving Decision. These instructions should contradict the provided future driving behavior of the ego driver. For example, extreme acceleration or requesting the driver to stop in the middle of the road while other vehicles are moving behind the ego driver. Include diverse reckless instructions with reasoning to reject them that include the other agents relative locations. These should have a Decision label of "<Rejected>."
- Task4: You must include 1 example of Instruction-Reasoning-Decision pairs that are not feasible due to a non feasible instruction even though it is not reckless driving. Should have a Decision label of "<Rejected>". Should be based on an actual thing that cannot be done with the current scenario due to surrounding objects relative locations. 
- Task5: You must include 1 example of Instruction-Reasoning-Decision with a random instruction that is not about driving. The reasoning should be, "As an AI Driver, what you provided is out of my scope," and the Decision should be "<Out_of_domain>."

Example outputs of each task (not based on the provided data):
- Task1 example: {{"Instruction": "Turn right, by first moving straight, then turn right at a slow speed", "Reasoning": "I can Turn right, by first moving straight, then turn right at a slow speed. By safely taking a right turn first, decelerating on the road, then turning right at a slow speed. I can do this because the traffic light is green, and no vehicles are in front of me or within high-risk proximity. Agent-2 is within a safe proximity behind.", "Decision": "<Accepted>"}}
- Task1 example: {{"Instruction": "Move straight with a fast speed, by first moving with a moderate speed then a high speed", "Reasoning": "I can move straight with a fast speed, by first moving with a moderate speed then a high speed, by safely starting to move and accelerating promptly going from a moderate speed to fast speed. This can be executed because there are no traffic lights and no vehicles or pedestrians within proximity. Agent-2 is on the right and moving with a slow speed.", "Decision": "<Accepted>"}}
- Task2 example: {{"Instruction": "Move straight while increasing the speed to 40 km/h", "Reasoning": "I can move straight while increasing the speed to 40 km/h, by cautiously starting to move straight with a low speed and then gently accelerate to change my speed from low to moderate. I will first accelerate gently because there is a car in the front-right of us driving at a slow speed. Once we pass the car, I will accelerate. We need to be cautious of the vehicle Agent-2 on the right that is traveling at a slower speed when accelerating.", "Decision": "<Accepted>"}}
- Task2 example: {{"Instruction": "Take a right turn by first turning right then moving straight, get closer to the vehicle in front", "Reasoning": "I can take a right turn by first turning right then moving straight, I will turn right with a low speed then gradually accelerate to match the speed of the vehicle Agent-2 on front and stay behind it in a safe distance.", "Decision": "<Accepted>"}}
- Task3 example: {{"Instruction": "Slow down to a speed of 2 km/h", "Reasoning": "Sorry, I cannot decelerate this aggressively due to vehicles behind moving at high speed, more specifically the closest vehicle Agent-2.", "Decision": "<Rejected>"}}
- Task4 example: {{"Instruction": "Move straight with a moderate speed", "Reasoning": "Sorry, I cannot move straight, there is a red traffic light just in front of us.", "Decision": "<Rejected>"}}
- Task4 example: {{"Instruction": "Move straight at a fast speed", "Reasoning": "Sorry, I cannot drive very fast; there are a lot of vehicles including Agent-2 in proximity and traveling at low speeds. Traveling fast will be a very high risk and could cause harm.", "Decision": "<Rejected>"}}
- Task5 example: {{"Instruction": "What time of day is it?", "Reasoning": "As an AI Driver, what you provided is out of my scope.", "Decision": "<Out_of_domain>"}}
    
"""

def get_ego_prompt(ego_agent):
    return f"""List of current and past ego vehicle states (can be used for reasoning of all tasks)
- Current location: {ego_agent['Current Location']}
- Current heading: {ego_agent['Current Heading']}
- Current speed: {ego_agent['Current Speed']}
- Interactive agent of interest: Agent-2

List of risk zones based on maximum observed speed of the ego vehicle (can be used for reasoning of all tasks)
- Extreme high risk zone range: {ego_agent['Distance in 1s']}
- High risk zone range: {ego_agent['Distance in 3s']}
- Safe zone range but within proximity: {ego_agent['Distance in 8s']}
If extreme risk zone is less than 3m, consider it as 3m. If high risk zone is less than 10m, consider it as 10m. If safe zone is less than 20m, consider it as 20m.

List of ego vehicle future direction and speed information (The only information used for the Instruction of Task1. Can be used for the Instruction of Task2. Should not be used for Task3 or Task4)
- The overall primary direction: {ego_agent['direction 0.1to8']} (Should be used in the instruction of Task1 and Task2)
- Step by step execution of the overall primary direction: first ({ego_agent['direction 0.1to4']}) then ({ego_agent['direction 4to8']})
- The overall primary speed category: {ego_agent['speed 0.1to8']} 
- Step by step execution of the overall primary speed category: first ({ego_agent['speed 0.1to4']}) then ({ego_agent['speed 4to8']})
- Step by step actual speed: The ego will reach a speed of {ego_agent['speed 4']} then a reach a speed of {ego_agent['speed 8']}

List of ego vehicle future locations (can be used for Instruction of Task2, the Reasoning only of Task1, should not be used for Task3 or Task4)
- Location after 4 seconds: {ego_agent['Future Location in 4s']}
- Heading after 4 seconds: {ego_agent['Future Heading in 4s']}
- Location after 8 seconds: {ego_agent['Future Location in 8s']}
- Heading after 8 seconds: {ego_agent['Future Heading in 8s']}

List of ego vehicle future acceleration and movement (no acceleration or movement information used for the Instruction of Task1 and Task2, can be used for Reasoning only of Task1 and Task2, should not be used for Task3 or Task4)
- The overall primary acceleration category the ego: {ego_agent['acceleration 0.1to8']} 
- Step by step acceleration: first ({ego_agent['acceleration 0.1to4']}) then ({ego_agent['acceleration 4to8']})
- Overall primary category of motion: ego_agent['movement 0.1to8']
- Step by step motion: first ({ego_agent['movement 0.1to4']}) then ({ego_agent['movement 4to8']})

"""

def get_distance_and_angle(location_str):
    current_location = np.array([float(i) for i in ''.join(''.join(location_str.split('(')).split(')')).split(', ')])
    current_distance = np.linalg.norm(current_location).round(2)
    current_angle = int(np.degrees(np.arctan2(current_location[1], current_location[0])))
    return current_distance, current_angle

def get_closest_traffic_prompt(map):
    traffic_prompt = "List of stop signs and traffic lights current states if they exist (can be used for reasoning of all tasks, can be combined with the Instruction of Task2 or Task3 or Task4)\n"
    no_traffic_light = True
    no_stop_sign = True
    # return_dict = "There are no stop signs.", "There are no traffic lights."
    if len(map)==0:
        return "There are no stop signs.", "There are no traffic lights."
    
    for map_k in map:
        if 'Stop' in map_k:
            no_stop_sign = False
        if 'Traffic' in map_k:
            no_traffic_light = False
    
    if no_stop_sign and no_traffic_light:
        return "There are no stop signs.", "There are no traffic lights."

    if no_stop_sign:
        stop_sign_str = "There are no stop signs."
    if no_traffic_light:
        traffic_light_str = "There are no traffic lights."
    
    list_of_traffic_lights = []
    list_of_traffic_lights_states = []
    list_of_stop_signs = []
    for map_k in map:
        if 'location' in map_k:
            distance, angle = get_distance_and_angle(map[map_k])
            if 'Traffic' in map_k:
                list_of_traffic_lights.append([distance, angle])
            else:
                list_of_stop_signs.append([distance, angle])
        else:
            list_of_traffic_lights_states.append(map[map_k])
        #     traffic_prompt+= f"- {map_k.replace('location', 'distance from ego')}: {distance} m\n"
        #     traffic_prompt+= f"- {map_k.replace('location', 'angle from ego')}: {angle} degrees\n"
        # else:
        #     traffic_prompt+= f"- {map_k}: {map[map_k]}\n"
    
    ahead_threshold_angle = 65

    if len(list_of_stop_signs)==0:
        stop_sign_str = "There are no stop signs."
    else:
        closest_idx = np.argmin(np.array(list_of_stop_signs)[:,0])
        closest_distance = list_of_stop_signs[closest_idx][0]
        closest_direction = "in front" if (abs(list_of_stop_signs[closest_idx][1])<ahead_threshold_angle) else "but not in front."
        stop_sign_str = f"The closest stop sign is {closest_distance}m {closest_direction}"

    if len(list_of_traffic_lights)==0:
        traffic_light_str = "There are no traffic lights."
    else:
        closest_idx = np.argmin(np.array(list_of_traffic_lights)[:,0])
        closest_distance = list_of_traffic_lights[closest_idx][0]
        closest_direction = "in front" if (abs(list_of_traffic_lights[closest_idx][1])<ahead_threshold_angle) else "but not in front."
        traffic_light_str = f"The closest traffic light ({list_of_traffic_lights_states[closest_idx]}) is {closest_distance}m {closest_direction}"
        
    return stop_sign_str, traffic_light_str

def get_traffic_prompt(map):
    traffic_prompt = "List of stop signs and traffic lights current states if they exist (can be used for reasoning of all tasks, can be combined with the Instruction of Task2 or Task3 or Task4)\n"
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
    return f"""List of current and past {i} states (can be used for reasoning of all tasks, can be used for the Instructions of Task2)
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
    
    prompt = get_context()
    # get_ego_prompt(agent['Agent-1'])
    prompt = prompt + get_ego_prompt(agent['Agent-1'])
    for agent_i in agent:
        if agent_i != 'Agent-1':
            prompt = prompt + get_agent_prompt(agent[agent_i], agent_i)

    prompt = prompt + get_traffic_prompt(map)

    return prompt

def generate_template_json(agent ,map=None, possible_directions=None, not_possible_directions=None, direction_classes=None):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    # """
    """{"Instruction": "Turn right, by first moving straight, then turn right at a slow speed", "Reasoning": "To execute the turn right instruction, the ego driver will first move straight to approach the turning point and then turn right at a slow speed to make a safe and smooth maneuver. This can be done as there are no high-risk objects within proximity and the interactive agent of interest, Agent-2, is at a safe distance on the right.", "Decision": "<Accepted>"}
{"Instruction": "Move straight while increasing the speed gradually to 24 km/h", "Reasoning": "The ego driver will smoothly move straight while increasing the speed gradually to 24 km/h. By cautiously accelerating, it will match the speed of surrounding agents, like Agent-4 and Agent-5. This can be safely executed as the speed increase is moderate and the surrounding objects are not in a high-risk zone.", "Decision": "<Accepted>"}
{"Instruction": "Drive at an extremely fast speed", "Reasoning": "Driving at an extremely fast speed is not feasible as it poses extreme risks to the safety of all objects in proximity, including Agent-3, Agent-10, and pedestrians like Agent-9. The ego driver should avoid this reckless behavior and maintain a safe speed considering the traffic environment.", "Decision": "<Rejected>"}
{"Instruction": "Make a sharp left turn", "Reasoning": "A sharp left turn is not feasible as there is no space for such a maneuver, and it could lead to a collision with Agent-20 that is directly ahead. The ego driver should prioritize safety by avoiding such risky and reckless maneuvers.", "Decision": "<Rejected>"}
{"Instruction": "What time is it now?", "Reasoning": "As an AI Driver, what you provided is out of my scope.", "Decision": "<Out_of_domain>"}
"""

    template_instruct = "You are generating the future motion plan for the ego vehicle. Reason about and predict the future multimodal trajectory embeddings of two agents—the ego (Agent-1) and another agent it interacts with (Agent-2) —based on the observed scene embeddings and the following ego instruction: Make the ego vehicle "
    
    direction = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to8'] else agent['Agent-1']['direction 0.1to8']
    direction1 = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to4'] else agent['Agent-1']['direction 0.1to4']
    direction2 = "stay stationary" if "stationary" in agent['Agent-1']['direction 4to8'] else agent['Agent-1']['direction 4to8']

    speed1 = agent['Agent-1']['speed 0.1to4']
    speed2 = agent['Agent-1']['speed 4to8']
    speed1 = f" with a {speed1} speed" if ("stationary" != speed1 and "INVALID" != speed1) else ""
    speed2 = f" with a {speed2} speed" if ("stationary" != speed2 and "INVALID" != speed2) else ""
    accel1 = agent['Agent-1']['acceleration 0.1to4']
    accel1 = f" and a {accel1}" if ("stationary" != accel1) else ""
    accel2 = agent['Agent-1']['acceleration 4to8']
    accel2 = f" and a {accel2}" if ("stationary" != accel2) else ""

    instruct1 = template_instruct + f"{direction}."
    instruct2 = template_instruct + f"{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}."
    instruct3 = template_instruct + f"{direction}, where it will first {direction1}, then {direction2}."
    action_reason = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}."
    
    # positive but no ground truth
    possible_not_ground_truth_directions=[]
    if possible_directions is not None:
        if len(possible_directions)>0:
            positive_directions = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in possible_directions]
            possible_not_ground_truth_directions = positive_directions # checking that it does not match the ground truth is already done outside this function, and thus shold not be checked here
            ## old bug found in the code, where if part of the ground truth action match the other feasible action it is removed from the other feasible set.
            # possible_not_ground_truth_directions = [direction_i for direction_i in positive_directions if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2)]
    positive_not_gt_instruct = []
    positive_not_gt_reason = []
    positive_not_gt_cls = []
    for direction_i in possible_not_ground_truth_directions:
        positive_not_gt_instruct.append(template_instruct + f"{direction_i}.")
        positive_not_gt_reason.append(f"The ego vehicle can {direction_i}.")
        if 'stationary' in direction_i:
            cls = np.where('stationary' == direction_classes)[0][0]
        else:
            cls = np.where(direction_i == direction_classes)[0][0]
        positive_not_gt_cls.append(cls)
    
    # negative
    negative_directions=[]
    if not_possible_directions is not None:
        if len(not_possible_directions)>0:
            negative_directions = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in not_possible_directions]
    negative_instruct = []
    negative_reason = []
    negative_cls = []
    for direction_i in negative_directions:
        # if 'stationary' not in direction_i and (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2):
        if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2): ## if it match the first or second step or the overall step direction of ground truth it should be skipped
            negative_instruct.append(template_instruct + f"{direction_i}.")
            negative_reason.append(f"The ego vehicle cannot {direction_i}, because there are no lanes that allow this direction based on the current location, heading, and speed of the ego vehicle.")
            if 'stationary' in direction_i:
                cls = np.where('stationary' == direction_classes)[0][0]
            else:
                cls = np.where(direction_i == direction_classes)[0][0]
            negative_cls.append(cls)
    #
    if 'Agent-2' in agent.keys():
        interactive_agent = agent['Agent-2']
        if 'Current Location' in interactive_agent.keys():
            current_distance,current_angle = get_distance_and_angle(interactive_agent['Current Location'])
            if abs(current_angle)<25:
                relative_location = 'in front'
            elif current_angle<120 and current_angle>25:
                relative_location = 'on the left'
            elif current_angle>-120 and current_angle<-25:
                relative_location = 'on the right'
            else:
                relative_location = 'behind'
        else:
            current_distance = 'unknown'
            relative_location = 'unknown'
        agent_2_reason = f'Agent-2 is {current_distance}m far {relative_location}.'
    else:
        agent_2_reason = ''
    
    if map is not None:
        stop_str, traffic_str = get_closest_traffic_prompt(map)
    else:
        stop_str, traffic_str = "", ""
    
    reason = action_reason
    reason2 = action_reason + " " + agent_2_reason + " " + traffic_str + " " + stop_str

    agent_type = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']

    xx = [str({"Instruction": instruct1, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct1, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n"]

    xx = xx + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Accepted>", "Label":"possible direction not gt", "Direction_cls": cls_i}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(positive_not_gt_instruct, positive_not_gt_reason, positive_not_gt_cls)]
    xx = xx + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Rejected>", "Label":"negative", "Direction_cls": cls_i}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(negative_instruct, negative_reason, negative_cls)]
    
    return "".join(xx)

def generate_template_json_2agent(agent ,map=None, possible_directions=None, not_possible_directions=None, direction_classes=None):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    # """
    """{"Instruction": "Turn right, by first moving straight, then turn right at a slow speed", "Reasoning": "To execute the turn right instruction, the ego driver will first move straight to approach the turning point and then turn right at a slow speed to make a safe and smooth maneuver. This can be done as there are no high-risk objects within proximity and the interactive agent of interest, Agent-2, is at a safe distance on the right.", "Decision": "<Accepted>"}
{"Instruction": "Move straight while increasing the speed gradually to 24 km/h", "Reasoning": "The ego driver will smoothly move straight while increasing the speed gradually to 24 km/h. By cautiously accelerating, it will match the speed of surrounding agents, like Agent-4 and Agent-5. This can be safely executed as the speed increase is moderate and the surrounding objects are not in a high-risk zone.", "Decision": "<Accepted>"}
{"Instruction": "Drive at an extremely fast speed", "Reasoning": "Driving at an extremely fast speed is not feasible as it poses extreme risks to the safety of all objects in proximity, including Agent-3, Agent-10, and pedestrians like Agent-9. The ego driver should avoid this reckless behavior and maintain a safe speed considering the traffic environment.", "Decision": "<Rejected>"}
{"Instruction": "Make a sharp left turn", "Reasoning": "A sharp left turn is not feasible as there is no space for such a maneuver, and it could lead to a collision with Agent-20 that is directly ahead. The ego driver should prioritize safety by avoiding such risky and reckless maneuvers.", "Decision": "<Rejected>"}
{"Instruction": "What time is it now?", "Reasoning": "As an AI Driver, what you provided is out of my scope.", "Decision": "<Out_of_domain>"}
"""
    # The same function, but with two agents instructions (Agent-1 and Agent-2)
    template_instruct = "You are generating the future motion plan for the ego vehicle. Reason about and predict the future multimodal trajectory embeddings of two agents—the ego (Agent-1) and another agent it interacts with (Agent-2) —based on the observed scene embeddings and the following ego instruction: Make the ego vehicle "
    
    direction = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to8'] else agent['Agent-1']['direction 0.1to8']
    direction1 = "stay stationary" if "stationary" in agent['Agent-1']['direction 0.1to4'] else agent['Agent-1']['direction 0.1to4']
    direction2 = "stay stationary" if "stationary" in agent['Agent-1']['direction 4to8'] else agent['Agent-1']['direction 4to8']

    direction_a2 = "stay stationary" if "stationary" in agent['Agent-2']['direction 0.1to8'] else agent['Agent-2']['direction 0.1to8']
    direction1_a2 = "stay stationary" if "stationary" in agent['Agent-2']['direction 0.1to4'] else agent['Agent-2']['direction 0.1to4']
    direction2_a2 = "stay stationary" if "stationary" in agent['Agent-2']['direction 4to8'] else agent['Agent-2']['direction 4to8']

    speed1 = agent['Agent-1']['speed 0.1to4']
    speed2 = agent['Agent-1']['speed 4to8']
    speed1 = f" with a {speed1} speed" if ("stationary" != speed1 and "INVALID" != speed1) else ""
    speed2 = f" with a {speed2} speed" if ("stationary" != speed2 and "INVALID" != speed2) else ""
    accel1 = agent['Agent-1']['acceleration 0.1to4']
    accel1 = f" and a {accel1}" if ("stationary" != accel1) else ""
    accel2 = agent['Agent-1']['acceleration 4to8']
    accel2 = f" and a {accel2}" if ("stationary" != accel2) else ""

    speed1_a2 = agent['Agent-2']['speed 0.1to4']
    speed2_a2 = agent['Agent-2']['speed 4to8']
    speed1_a2 = f" with a {speed1_a2} speed" if ("stationary" != speed1_a2 and "INVALID" != speed1_a2) else ""
    speed2_a2 = f" with a {speed2_a2} speed" if ("stationary" != speed2_a2 and "INVALID" != speed2_a2) else ""
    accel1_a2 = agent['Agent-2']['acceleration 0.1to4']
    accel1_a2 = f" and a {accel1_a2}" if ("stationary" != accel1_a2) else ""
    accel2_a2 = agent['Agent-2']['acceleration 4to8']
    accel2_a2 = f" and a {accel2_a2}" if ("stationary" != accel2_a2) else ""

    instruct1 = template_instruct + f"{direction}."
    instruct2 = template_instruct + f"{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}."
    instruct3 = template_instruct + f"{direction}, where it will first {direction1}, then {direction2}."
    action_reason = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}."

    instruct1_a2 = template_instruct + f"{direction} while Agent-2 will {direction_a2}."
    instruct2_a2 = template_instruct + f"{direction}, where it will first {direction1}{speed1}, then {direction2}{speed2}, while Agent-2 will first {direction1_a2}{speed1_a2}, then {direction2_a2}{speed2_a2}."
    instruct3_a2 = template_instruct + f"{direction}, where it will first {direction1}, then {direction2}, while Agent-2 will first {direction1_a2}, then {direction2_a2}."
    action_reason_a2 = f"The ego vehicle can {direction}, where it will first {direction1}{speed1}{accel1}, then {direction2}{speed2}{accel2}, while Agent-2 will {direction_a2}, where it will first {direction1_a2}{speed1_a2}{accel1_a2}, then {direction2_a2}{speed2_a2}{accel2_a2}."
    
    # positive but no ground truth
    possible_not_ground_truth_directions=[]
    if possible_directions is not None:
        if len(possible_directions)>0:
            positive_directions = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in possible_directions]
            possible_not_ground_truth_directions = positive_directions # checking that it does not match the ground truth is already done outside this function, and thus shold not be checked here
            ## old bug found in the code, where if part of the ground truth action match the other feasible action it is removed from the other feasible set.
            # possible_not_ground_truth_directions = [direction_i for direction_i in positive_directions if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2)]
    positive_not_gt_instruct = []
    positive_not_gt_reason = []
    positive_not_gt_cls = []
    for direction_i in possible_not_ground_truth_directions:
        positive_not_gt_instruct.append(template_instruct + f"{direction_i}.")
        positive_not_gt_reason.append(f"The ego vehicle can {direction_i}.")
        if 'stationary' in direction_i:
            cls = np.where('stationary' == direction_classes)[0][0]
        else:
            cls = np.where(direction_i == direction_classes)[0][0]
        positive_not_gt_cls.append(cls)
    
    # positive but no ground truth, a2
    possible_not_ground_truth_directions_a2=[]
    if possible_directions is not None:
        if len(possible_directions)>0:
            positive_directions = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in possible_directions]
            possible_not_ground_truth_directions_a2 = positive_directions # checking that it does not match the ground truth is already done outside this function, and thus shold not be checked here
            ## old bug found in the code, where if part of the ground truth action match the other feasible action it is removed from the other feasible set.
            # possible_not_ground_truth_directions = [direction_i for direction_i in positive_directions if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2)]
    positive_not_gt_instruct_a2 = []
    positive_not_gt_reason_a2 = []
    positive_not_gt_cls_a2 = []
    for direction_i in possible_not_ground_truth_directions_a2:
        positive_not_gt_instruct_a2.append(template_instruct + f"{direction_i} while Agent-2 will {direction_a2}.")
        positive_not_gt_reason_a2.append(f"The ego vehicle can {direction_i}, while Agent-2 will {direction_a2}.")
        if 'stationary' in direction_i:
            cls = np.where('stationary' == direction_classes)[0][0]
        else:
            cls = np.where(direction_i == direction_classes)[0][0]
        positive_not_gt_cls_a2.append(cls)


    # negative
    negative_directions=[]
    if not_possible_directions is not None:
        if len(not_possible_directions)>0:
            negative_directions = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in not_possible_directions]
    negative_instruct = []
    negative_reason = []
    negative_cls = []
    for direction_i in negative_directions:
        # if 'stationary' not in direction_i and (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2):
        if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2): ## if it match the first or second step or the overall step direction of ground truth it should be skipped
            negative_instruct.append(template_instruct + f"{direction_i}.")
            negative_reason.append(f"The ego vehicle cannot {direction_i}, because there are no lanes that allow this direction based on the current location, heading, and speed of the ego vehicle.")
            if 'stationary' in direction_i:
                cls = np.where('stationary' == direction_classes)[0][0]
            else:
                cls = np.where(direction_i == direction_classes)[0][0]
            negative_cls.append(cls)

    # negative a2
    negative_directions_a2=[]
    if not_possible_directions is not None:
        if len(not_possible_directions)>0:
            negative_directions_a2 = ["stay stationary" if "stationary" in direction_i else direction_i for direction_i in not_possible_directions]
    negative_instruct_a2 = []
    negative_reason_a2 = []
    negative_cls_a2 = []
    for direction_i in negative_directions_a2:
        # if 'stationary' not in direction_i and (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2):
        if (direction_i!=direction and direction_i!=direction1 and direction_i!=direction2): ## if it match the first or second step or the overall step direction of ground truth it should be skipped
            negative_instruct_a2.append(template_instruct + f"{direction_i} while Agent-2 will {direction_a2}.")
            negative_reason_a2.append(f"The ego vehicle cannot {direction_i}, because there are no lanes that allow this direction based on the current location, heading, and speed of the ego vehicle, while Agent-2 will {direction_a2}.")
            if 'stationary' in direction_i:
                cls = np.where('stationary' == direction_classes)[0][0]
            else:
                cls = np.where(direction_i == direction_classes)[0][0]
            negative_cls_a2.append(cls)


    if 'Agent-2' in agent.keys():
        interactive_agent = agent['Agent-2']
        if 'Current Location' in interactive_agent.keys():
            current_distance,current_angle = get_distance_and_angle(interactive_agent['Current Location'])
            if abs(current_angle)<25:
                relative_location = 'in front'
            elif current_angle<120 and current_angle>25:
                relative_location = 'on the left'
            elif current_angle>-120 and current_angle<-25:
                relative_location = 'on the right'
            else:
                relative_location = 'behind'
        else:
            current_distance = 'unknown'
            relative_location = 'unknown'
        agent_2_reason = f'Agent-2 is {current_distance}m far {relative_location}.'
    else:
        agent_2_reason = ''
    
    if map is not None:
        stop_str, traffic_str = get_closest_traffic_prompt(map)
    else:
        stop_str, traffic_str = "", ""
    
    reason = action_reason
    reason2 = action_reason + " " + agent_2_reason + " " + traffic_str + " " + stop_str

    reason_a2 = action_reason_a2
    reason2_a2 = action_reason_a2 + " " + agent_2_reason + " " + traffic_str + " "+ stop_str

    agent_type = ['Unknown', 'Vehicle', 'Pedestrian', 'Cyclist']

    
    xx = [str({"Instruction": instruct1, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3, "Reasoning": reason, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct1, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3, "Reasoning": reason2, "Decision":"<Accepted>", "Label":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls']}).replace("'", '"')+"\n"]

    xx = xx + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Accepted>", "Label":"possible direction not gt", "Direction_cls": cls_i}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(positive_not_gt_instruct, positive_not_gt_reason, positive_not_gt_cls)]
    xx = xx + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Rejected>", "Label":"negative", "Direction_cls": cls_i}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(negative_instruct, negative_reason, negative_cls)]
    
    # Similar to the above replace all the single agent data with the _a2 data, and inlcude the corresponding lable, direction_cls and so on similarly
    xx_a2 = [str({"Instruction": instruct1_a2, "Reasoning": reason_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2_a2, "Reasoning": reason_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3_a2, "Reasoning": reason_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct1_a2, "Reasoning": reason2_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct2_a2, "Reasoning": reason2_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n",
    str({"Instruction": instruct3_a2, "Reasoning": reason2_a2, "Decision":"<Accepted>", "Label":"gt", "Label_a2":"gt", "Direction_cls": agent['Agent-1']['direction 0.1to8_cls'], "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n"]

    xx_a2 = xx_a2 + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Accepted>", "Label":"possible direction not gt", "Label_a2":"gt", "Direction_cls": cls_i, "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(positive_not_gt_instruct_a2, positive_not_gt_reason_a2, positive_not_gt_cls_a2)]
    xx_a2 = xx_a2 + [str({"Instruction": instruct_i, "Reasoning": reason_i, "Decision":"<Rejected>", "Label":"negative", "Label_a2":"gt", "Direction_cls": cls_i, "Direction_cls_a2": agent['Agent-2']['direction 0.1to8_cls']}).replace("'", '"')+"\n" for instruct_i, reason_i, cls_i in zip(negative_instruct_a2, negative_reason_a2, negative_cls_a2)]
    
    return "".join(xx_a2)
    # return "".join(xx)

def get_input_data(agent ,map):
    """
        Given a dict contains all the info about the scene,
        such as the ego vehicle status, traffic lights, stop signs, etc..
        will generate the meta/template prompt.
    # """
    
    # prompt = get_context()
    prompt = ''
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
    
    # gpt_prompt = generate_meta_caption(agent_json, map_json)
    gpt_prompt = generate_template_json(agent_json, map_json)
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


        