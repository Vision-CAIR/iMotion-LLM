import os
import json
import time
import openai
from pathlib import Path
import pickle
import glob
from tqdm import tqdm

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise ValueError("Set OPENAI_API_KEY before running this script.")
openai.api_key = api_key
# Load the GPT grading prompt template
REPO_ROOT = Path(__file__).resolve().parents[2]
GRADING_PROMPT_PATH = REPO_ROOT / "gameformer" / "nuplan_preprocess" / "gpt_grader.txt"
# GROUND_TRUTH_JSON_PATH = "/ibex/project/c2278/felembaa/datasets/nuplan/test_gpt_prompt_14types/all_types/gpt_data_101124/us-pa-pittsburgh-hazelwood_cda268e846265f33.json"

# Output directory
# OUTPUT_JSON_PATH = "/ibex/project/c2278/felembaa/datasets/nuplan/test_gpt_prompt_14types/all_types/gpt_grading_results.json"

def load_file(file_path):
    """ Load a text file and return its content. """
    with open(file_path, "r") as f:
        return f.read().strip()

def load_json(file_path):
    """ Load a JSON file and return its content. """
    with open(file_path, "r") as f:
        return json.load(f)


def load_pkl(file_path):
    """ Load a Pickle (.pkl) file and return its content. """
    with open(file_path, "rb") as f:  # Open in binary read mode
        return pickle.load(f)

def call_chatgpt_03(prompt, max_retries=3, delay=5, model="gpt-4o-mini"):
    """ Calls OpenAI GPT model with a given prompt, handling retries. """
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )
            return response['choices'][0]['message']['content']
        except openai.error.OpenAIError as e:
            print(f"Error occurred: {e}, retrying ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(delay)
    return None  # Return None if all retries fail

def generate_grading_prompt(template, given_instruction, answer, correct_answer):
    """ Fills the grading template with the given instruction, answer, and correct answer. """
    return template.replace("{Given_instruction}", given_instruction) \
                   .replace("{answer}", answer) \
                   .replace("{correct_answer}", correct_answer)

# def process_instruction(instruction_data, grading_template):
def process_instruction(instruction_text, correct_answer_text, answer_text, grading_template):
    """ Processes a single instruction and evaluates it using GPT grading. """
    # instruction_text = instruction_data["instruction"]
    # instruction_data["answer"] = instruction_data["reasoning"]
    # answer_text = instruction_data["answer"]
    # correct_answer_text = instruction_data["reasoning"]

    # Fill the grading prompt
    grading_prompt = generate_grading_prompt(grading_template, instruction_text, answer_text, correct_answer_text)

    # Call GPT for grading
    grading_response = call_chatgpt_03(grading_prompt)

    # Parse GPT output (ensure it's JSON formatted)
    try:
        grading_json = json.loads(grading_response.replace("```json", "").replace("```", ""))
    except json.JSONDecodeError:
        grading_json = {"score": 0, "justification": "Failed to parse GPT response", "errors": []}

    # Store the grading result with original instruction data
    return grading_json
    # instruction_data["grading"] = grading_json
    # return instruction_data
def save_pkl(file_path, data):
    """ Save a Python object to a Pickle (.pkl) file. """
    with open(file_path, "wb") as f:  # Write in binary mode
        pickle.dump(data, f)

def save_score_to_txt(score, count, save_path):
    """ Save the overall score to a text file. """
    score_text = f"Overall Score = {score/count:.4f}"  # Format with 4 decimal places
    with open(save_path, "w") as f:
        f.write(score_text)
    print(f"Saved overall score to {save_path}")

def main():
    """ Main function to process all instructions and evaluate them sequentially. """
    # Load grading prompt template
    graded_results = []
    grading_template = load_file(GRADING_PROMPT_PATH)
    evaluated_file = '/ibex/project/c2278/cvpr_rebuttal/complex/r8/checkpoint-3600/result/gt1_eval0/data/*'
    target_files = glob.glob(evaluated_file)
    scores_count, scores = 0, 0
    score_save_path = '/'.join(evaluated_file.split('/')[:-2])+'overall_score.txt'
    for file_i in tqdm(target_files):
        loaded_data = load_pkl(file_i)
        instruction = loaded_data['instruct']
        correct_answer = loaded_data['caption']
        answer = loaded_data['answer']
        if 'score' in loaded_data:
            # for k,v in {"Instruction": loaded_data['instruct'], "Correct Answer": loaded_data['caption'], "iMotion-LLM": loaded_data['answer']}.items():
            #     print(f"> {k}: \n{v}")
            scores_count += 1 
            scores += loaded_data['score']
            # print(f"overall score = {scores/scores_count}")
            # save_score_to_txt(scores, scores_count, score_save_path)
            # for k,v in graded_result.items():
            #     print(f"# {k}: \n{v}")
        else:
            graded_result = process_instruction(instruction, correct_answer, answer, grading_template)
            loaded_data.update(graded_result)
            # graded_result.update({"Instruction": loaded_data['instruct'], "Correct Answer": loaded_data['caption'], "iMotion-LLM": loaded_data['answer']})
            # graded_results.append(graded_result)
            for k,v in {"Instruction": loaded_data['instruct'], "Correct Answer": loaded_data['caption'], "iMotion-LLM": loaded_data['answer']}.items():
                print(f"> {k}: \n{v}")
            for k,v in graded_result.items():
                print(f"# {k}: \n{v}")
            # print(f"Grading result: {graded_result['grading']}")
            scores_count += 1 
            scores += graded_result['score']
            print(f"overall score = {scores/scores_count}")
            print("-" * 80)
            save_pkl(file_i, loaded_data)
            # save_score_to_txt(scores, scores_count, score_save_path)
    
    # print(f"overall score = {scores/scores_count}")
    # # graded_results.append({"overall_score": scores/scores_count})
    # # Save the graded results
    # with open(OUTPUT_JSON_PATH, "w") as output_file:
    #     json.dump(graded_results, output_file, indent=4)
    print(f"overall score = {scores/scores_count}")
    # Save to text file
    save_score_to_txt(scores, scores_count, score_save_path)
    print(f"Grading completed. Results saved to {score_save_path}")
    # # Load ground truth dataset
    # ground_truth_data = load_json(GROUND_TRUTH_JSON_PATH)

    # # Process each instruction sequentially (for debugging)
    # graded_results = []
    # scores = 0
    # scores_count = 0
    # for scenario in ground_truth_data:
    #     for instruction_set in scenario["data"]:
    #         for instruction in instruction_set["instructions"]:
    #             print(f"Processing instruction: {instruction['instruction']}")
    #             graded_result = process_instruction(instruction, grading_template)
    #             graded_results.append(graded_result)
    #             print(f"Grading result: {graded_result['grading']}")
    #             scores_count += 1 
    #             scores += graded_result['grading']['score']
    #             print("-" * 80)

    # print(f"overall score = {scores/scores_count}")
    # graded_results.append({"overall_score": scores/scores_count})
    # # Save the graded results
    # with open(OUTPUT_JSON_PATH, "w") as output_file:
    #     json.dump(graded_results, output_file, indent=4)

    # print(f"Grading completed. Results saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()





# ###############################
# import os
# import json
# import time
# import openai
# from pathlib import Path
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # Load the GPT grading prompt template
# GRADING_PROMPT_PATH = "/home/felembaa/projects/iMotion-LLM-ICLR/gameformer/nuplan_preprocess/gpt_grader.txt"
# GROUND_TRUTH_JSON_PATH = "/ibex/project/c2278/felembaa/datasets/nuplan/test_gpt_prompt_14types/all_types/gpt_data_101124/us-pa-pittsburgh-hazelwood_cda268e846265f33.json"

# # Output directory
# OUTPUT_JSON_PATH = "/ibex/project/c2278/felembaa/datasets/nuplan/test_gpt_prompt_14types/all_types/gpt_grading_results.json"

# def load_file(file_path):
#     """ Load a text file and return its content. """
#     with open(file_path, "r") as f:
#         return f.read().strip()

# def load_json(file_path):
#     """ Load a JSON file and return its content. """
#     with open(file_path, "r") as f:
#         return json.load(f)

# def call_chatgpt_03(prompt, max_retries=3, delay=5, model="gpt-4o-mini"):
#     """ Calls OpenAI GPT model with a given prompt, handling retries. """
#     retries = 0
#     while retries < max_retries:
#         try:
#             response = openai.ChatCompletion.create(
#                 model=model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt},
#                 ]
#             )
#             return response['choices'][0]['message']['content']
#         except openai.error.OpenAIError as e:
#             print(f"Error occurred: {e}, retrying ({retries + 1}/{max_retries})...")
#             retries += 1
#             time.sleep(delay)
#     return None  # Return None if all retries fail

# def generate_grading_prompt(template, given_instruction, answer, correct_answer):
#     """ Fills the grading template with the given instruction, answer, and correct answer. """
#     return template.replace("{Given_instruction}", given_instruction) \
#                    .replace("{answer}", answer) \
#                    .replace("{correct_answer}", correct_answer)

# def process_instruction(instruction_data, grading_template):
#     """ Processes a single instruction and evaluates it using GPT grading. """
#     instruction_text = instruction_data["instruction"]
#     answer_text = instruction_data["answer"]
#     correct_answer_text = instruction_data["reasoning"]

#     # Fill the grading prompt
#     grading_prompt = generate_grading_prompt(grading_template, instruction_text, answer_text, correct_answer_text)

#     # Call GPT for grading
#     grading_response = call_chatgpt_03(grading_prompt)

#     # Parse GPT output (ensure it's JSON formatted)
#     try:
#         grading_json = json.loads(grading_response.replace("```json", "").replace("```", ""))
#     except json.JSONDecodeError:
#         grading_json = {"score": 0, "justification": "Failed to parse GPT response", "errors": []}

#     # Store the grading result with original instruction data
#     instruction_data["grading"] = grading_json
#     return instruction_data

# def main():
#     """ Main function to process all instructions and evaluate them. """
#     # Load grading prompt template
#     grading_template = load_file(GRADING_PROMPT_PATH)

#     # Load ground truth dataset
#     ground_truth_data = load_json(GROUND_TRUTH_JSON_PATH)

#     # Process each instruction in parallel
#     graded_results = []
#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = [
#             executor.submit(process_instruction, instruction, grading_template)
#             for scenario in ground_truth_data
#             for instruction_set in scenario["data"]
#             for instruction in instruction_set["instructions"]
#         ]
#         for future in as_completed(futures):
#             graded_results.append(future.result())

#     # Save the graded results
#     with open(OUTPUT_JSON_PATH, "w") as output_file:
#         json.dump(graded_results, output_file, indent=4)

#     print(f"Grading completed. Results saved to {OUTPUT_JSON_PATH}")

# if __name__ == "__main__":
#     main()
