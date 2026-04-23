# pylint: skip-file
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import click
import openai
from retry import retry
from tqdm import tqdm
import sys
sys.path.append(".")
from chatgpt_instruct_v02 import *
from PIL import Image
import base64, requests
import copy
# from retrying import retry

REPO_ROOT = Path(__file__).resolve().parents[2]
REASONING_TEMPLATE_PATH = REPO_ROOT / "gameformer" / "nuplan_preprocess" / "prompts_templates" / "reasoning_template.txt"

# @retry(stop_max_attempt_number=3, wait_fixed=2000)
@retry(tries=5, delay=2, backoff=2)
def make_gpt_request(headers, payload):
    # Get the JSON response content
    # Make API request
    response = openai.Completion.create(
    model=payload['model'],
    prompt=payload['messages'][0]['content'],
    # response_format={ "type": "json_object" }
    # “response_format”: “json_object”
    )
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    output = response.json()
    output_text = output["choices"][0]["message"]["content"]
    # output_text = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()["choices"][0]["message"]["content"]
    json_data = json.loads(output_text.replace('json', '').replace("```", ''))
    return json_data

def make_gpt_request_02(user_input):
    prompt = user_input
    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data generation tool for self driving."},
            {"role": "user", "content": prompt},
        ],
        functions=[{"name": "dummy_fn", "parameters": { 
          "type": "object",
          "properties": { 
            "instructions": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "abstract instruction": {"type": "string", "description": "short abstract driving instruction"},
                  "detailed instruction": {"type": "string", "description": "detailed driving instruction"},
                  "safe": {"type": "boolean","description": "True or False to indicate safe or not."}
                }
              }
            }
          }
        }}],
    )
    try:
        generated_text = completion.choices[0].message.function_call.arguments
        return json.loads(generated_text)
    except Exception as e:
        print(f"An error occurred: {e}")
        return

def call_chatgpt_03(prompt, max_retries=3, delay=5, model="gpt-4o-mini"):
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,  # gpt-4o-mini, gpt-4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            answer = response['choices'][0]['message']['content']
            return answer  # Exit the function after a successful response

        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {delay} seconds... ({retries}/{max_retries})")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Exiting.")
                return None

@retry(tries=5, delay=2, backoff=2)
def make_description_from_prompt(input_prompt, context):
    
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo-0125",
        # model="gpt-3.5-turbo-0125",
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": input_prompt},
        ],
        temperature=1.0,
        # response_format={"type": "json_object"},
    )

    first_response = response["choices"][0]["message"]["content"]
    print(first_response)
    return first_response

def process_frame(frame_num, observation, input_prompt, cache_path):
    # if os.path.exists(cache_path):
    #     with open(cache_path, "rb") as f:
    #         result = pickle.load(f)
    #     print("loading from cache: ", frame_num)
    #     return result
    context = get_context()
    print("making description for frame: ", frame_num)
    response_content = make_description_from_prompt(input_prompt, context)
    result = {
        "frame_num": frame_num,
        "observation": observation,
        "input_prompt": input_prompt,
        "response_content": response_content,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result

def main(
    # input_path,
    # output_folder,
    # max_steps,
    # stride,
    # num_process,
    # openai_api,
):
    # Init openai api key
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running this script.")
    openai.api_key = api_key

    
    # filename = 'aab13ca1ad535214_20_31_interest'
    # root_dir = "<internal_dataset_root>/nuplan/temp_gpt"

    root_dir_ = "<internal_dataset_root>/nuplan/gpt_prompt_14types/*"
    root_dirs_ = glob.glob(root_dir_)
    root_dirs_ = [i+'/_gpt' for i in root_dirs_]
    root_dirs_ = ['<internal_dataset_root>/nuplan/gpt_prompt_14types/accelerating_at_crosswalk/gpt']
    for root_dir in root_dirs_:
        # root_dir = "<internal_dataset_root>/nuplan/gpt_prompt_sample_01_gpt"
        # save_dir = "<internal_dataset_root>/nuplan/gpt_prompt_sample_01_gpt_output_4o_mini"
        save_dir = root_dir+'_complex_instruct_reason_data'
        os.makedirs(save_dir, exist_ok=True)
        fig_dirs = f"{root_dir}_figures"
        prompt_dirs = f"{root_dir}_prompts"
        pos_dirs = f"{root_dir}_positive_prompts"
        neg_dirs = f"{root_dir}_negative_prompts"
        meta_dirs = f"{root_dir}_meta_prompts"
        # root_dir = '<internal_dataset_root>/waymo/gameformer/temp'
        # agent_dir = root_dir+'_agentJsons'
        # map_dir = root_dir+'_mapJsons'

        files_ = glob.glob(fig_dirs+'/*')
        for fig_filename in tqdm(files_):
            
            t1 = time.time()
        # fig_filename = files_[0]
            filename = fig_filename.split('/')[-1].replace('.png','')
            pos_filename = f"{pos_dirs}/{filename}.txt"
            neg_filename = f"{neg_dirs}/{filename}.txt"
            prompt_filename = f"{prompt_dirs}/{filename}.txt"
            meta_filename = f"{meta_dirs}/{filename}.txt"

            if os.path.exists(f"{save_dir}/{filename}.json"):
                continue
            

            loaded_pos_prompt, loaded_neg_prompt = (
                open(pos_filename, 'r').read(),
                open(neg_filename, 'r').read()
            )
            reasoning_template = open(REASONING_TEMPLATE_PATH, 'r').read()
            reasoning_template = reasoning_template.replace('driving_behaviour', 'trajectory_description')
            meta_data = json.loads(open(meta_filename, 'r').read())
            meta_data['scenario_type'] = meta_data['scenario_type'].replace('_', ' ')
            for k,v in meta_data.items():
                reasoning_template = reasoning_template.replace(f"<{k}>", v)
            # loaded_gpt_prompt = open(prompt_filename, 'r').read()
            # # Load image as binary for OpenAI API
            # with open(fig_filename, "rb") as image_file:
            #     loaded_fig = image_file.read()

            # Use ImageCompletion for multimodal input with GPT-4o mini
            prompt_text=loaded_pos_prompt
            
            # with open(fig_filename, "rb") as image_file:
            #     base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # set headers
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            gpt_output_jsons = []
            # for prompt_text in [loaded_pos_prompt, loaded_neg_prompt, loaded_gpt_prompt]:
            # for prompt_text in [loaded_gpt_prompt]:
            # for prompt_text in [loaded_pos_prompt]:
            for prompt_text in [loaded_pos_prompt, loaded_neg_prompt]:
            # for prompt_text in [loaded_neg_prompt]:
            # for prompt_text in [loaded_gpt_prompt]:
                # if prompt_text == loaded_neg_prompt and "here are few positive (safe) examples" in prompt_text and json_data is not None:
                #     prompt_text = prompt_text + str(json_data[0])+'\n'+str(json_data[3])+'\n'

                json_data = call_chatgpt_03(prompt_text, model="gpt-4o-mini")
                try:
                    # Attempt to parse the JSON
                    json_data = json.loads(json_data.replace('json', '').replace("```", ''))
                except json.JSONDecodeError:
                    # If parsing fails, ask ChatGPT to fix the JSON
                    fix_json_prompt = f"""The JSON data returned could not be parsed. Please provide the data again in valid JSON format, without any extra text or formatting.
Data to fix:
{json_data}
"""
                    # Call ChatGPT with the fix prompt
                    fixed_json_response = call_chatgpt_03(fix_json_prompt, model="gpt-4o-mini")
                    # Try parsing the fixed JSON
                    try:
                        json_data = json.loads(fixed_json_response.replace('json', '').replace("```", ''))
                    except json.JSONDecodeError:
                        json_data = None
                # json_data = make_gpt_request_02(prompt_text)
                # json_data = json_data if json_data is not None else make_gpt_request_02(prompt_text)
                if json_data is None:
                    continue
                # json_data = make_gpt_request(headers, payload)
                # reasoning:
                json_data_temp = copy.deepcopy(json_data)
                for ii in range(len(json_data_temp)):
                    for jj in range(len(json_data_temp[ii]['instructions'])):
                        json_data_temp[ii]['instructions'][jj].pop('safe')
                        json_data_temp[ii]['instructions'][jj].pop('category')
                
                json_data_with_reasoning = call_chatgpt_03(reasoning_template + str(json_data_temp), model="gpt-4o-mini")
                # Try parsing the fixed JSON
                try:
                    json_data_with_reasoning = json.loads(json_data_with_reasoning.replace('json', '').replace("```", ''))
                except json.JSONDecodeError:
                    json_data_with_reasoning = None
                if json_data_with_reasoning is None:
                    continue
                
                for ii in range(len(json_data)):
                    for jj in range(len(json_data[ii]['instructions'])):
                        json_data_with_reasoning[ii]['instructions'][jj]['safe_instruction'] = json_data[ii]['instructions'][jj]['safe']
                        json_data_with_reasoning[ii]['instructions'][jj]['category'] = json_data[ii]['instructions'][jj]['category']

                json_data_ = {"prompt": prompt_text, "data": json_data_with_reasoning}
                
                gpt_output_jsons.append(json_data_)
                # print()
                # for json_data_i in json_data:
                #     gpt_output_jsons.append(json_data_i)
            # Save to a JSON file
            print(f"time: {time.time() - t1}s")
            if json_data is None:
                continue
            # print(f"{save_dir}/{filename}.json")
            with open(f"{save_dir}/{filename}.json", "w") as json_file:
                json.dump(gpt_output_jsons, json_file, indent=4)  # indent=4 for pretty formatting
    # print('')


if __name__ == "__main__":
    main()
    # return response.choices[0].message.content
