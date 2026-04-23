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
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
# from retrying import retry

REPO_ROOT = Path(__file__).resolve().parents[2]
REASONING_TEMPLATE_PATH = REPO_ROOT / "gameformer" / "nuplan_preprocess" / "prompts_templates" / "reasoning_template.txt"


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

def process_file(fig_filename, root_dir, save_dir, api_key):
    filename = fig_filename.split('/')[-1].replace('.png', '')
    pos_filename = f"{root_dir}_positive_prompts/{filename}.txt"
    neg_filename = f"{root_dir}_negative_prompts/{filename}.txt"
    meta_filename = f"{root_dir}_meta_prompts/{filename}.txt"

    if os.path.exists(f"{save_dir}/{filename}.json"):
        return f"Skipping {filename}, output already exists."

    loaded_pos_prompt = open(pos_filename, 'r').read()
    loaded_neg_prompt = open(neg_filename, 'r').read()
    reasoning_template = open(REASONING_TEMPLATE_PATH, 'r').read().replace('driving_behaviour', 'trajectory_description')
    
    meta_data = json.loads(open(meta_filename, 'r').read())
    for k, v in meta_data.items():
        reasoning_template = reasoning_template.replace(f"<{k}>", v)

    gpt_output_jsons = []
    for prompt_text in [loaded_pos_prompt, loaded_neg_prompt]:
        json_data = call_chatgpt_03(prompt_text, model="gpt-4o-mini")
        try:
            json_data = json.loads(json_data.replace('json', '').replace("```", ''))
        except json.JSONDecodeError:
            json_data = None

        if json_data is None:
            continue

        json_data_temp = copy.deepcopy(json_data)
        for ii in range(len(json_data_temp)):
            for jj in range(len(json_data_temp[ii]['instructions'])):
                json_data_temp[ii]['instructions'][jj].pop('safe', None)
                json_data_temp[ii]['instructions'][jj].pop('category', None)

        json_data_with_reasoning = call_chatgpt_03(reasoning_template + str(json_data_temp), model="gpt-4o-mini")
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

    with open(f"{save_dir}/{filename}.json", "w") as json_file:
        json.dump(gpt_output_jsons, json_file, indent=4)
    
    return f"Processed {filename}"

def main():
    parser = argparse.ArgumentParser(description='Process a counter.')
    parser.add_argument('--type_select', type=int, help='Number of splits', default=0)
    parser.add_argument('--processes', type=int, help='number of workers', default=1)
    args = parser.parse_args()

    root_dir_ = "/ibex/project/c2278/felembaa/datasets/nuplan/gpt_prompt_14types/*"
    root_dirs_ = glob.glob(root_dir_)
    root_dir = root_dirs_[args.type_select]+'/gpt'  # Select the directory
    save_dir = root_dir + '_data_101124'
    os.makedirs(save_dir, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY before running this script.")
    openai.api_key = api_key

    fig_dirs = f"{root_dir}_figures"
    files_ = glob.glob(fig_dirs + '/*')

    with ProcessPoolExecutor(max_workers=args.processes) as executor:
        futures = {executor.submit(process_file, fig_filename, root_dir, save_dir, api_key): fig_filename for fig_filename in files_}
        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
    # return response.choices[0].message.content
