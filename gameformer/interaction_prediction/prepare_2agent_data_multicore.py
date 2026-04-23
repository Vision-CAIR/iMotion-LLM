import os
import glob
import json
import shutil
from tqdm import tqdm
from multiprocessing import Pool, Manager

root_dir = '/ibex/project/c2278/felembaa/datasets/waymo/gameformer/'

train_dir_full = 'training_full_3jul'
train_dir_small = 'training_small_1jul'
valid_dir = 'validation_3jul'

train_full_data_list = glob.glob(root_dir + train_dir_full + '/*')
print(len(train_full_data_list), flush=True)
filenames = train_full_data_list
output_dir = root_dir + train_dir_full + '_filenames_T/filenames_T.json'
output_root_dir = root_dir + train_dir_full

filename_T_dict = {}
filenames_str = ';'.join(filenames)
long_string = filenames_str

def process_filename_range(indices, long_string, output_root_dir):
    partial_dict = {}
    for index in indices:
        filename = filenames[index]
        start_substring = '_'.join([filename.split('/')[-1].split('_')[0], filename.split('/')[-1].split('_')[2]])
        start_index = long_string.find(start_substring)
        end_substring = '.npz'
        if start_index != -1:
            end_index = long_string.find(end_substring, start_index) + len(end_substring)
            if end_index != -1:
                partial_dict[filename] = output_root_dir + '/' + long_string[start_index:end_index]
    return partial_dict

def update_progress(result):
    pbar.update()

# Splitting filenames into chunks for parallel processing
num_processes = 100
split_indices = [list(range(i, min(i + num_processes, len(filenames)))) for i in range(0, len(filenames), num_processes)]

manager = Manager()
pbar = tqdm(total=len(split_indices), desc="Processing files", unit="chunk")

# Process files in parallel
with Pool(processes=num_processes) as pool:
    results = []
    for indices in split_indices:
        result = pool.apply_async(process_filename_range, args=(indices, long_string, output_root_dir), callback=update_progress)
        results.append(result)
    
    pool.close()
    pool.join()

# Merge partial results into filename_T_dict
for result in results:
    partial_dict = result.get()
    filename_T_dict.update(partial_dict)

# Ensure the output directory exists and write the result to the JSON file
ff = '/'.join(output_dir.split('/')[:-1])
if os.path.exists(ff):
    shutil.rmtree(ff)
os.makedirs(ff, exist_ok=True)
with open(output_dir, 'w') as file:
    json.dump(filename_T_dict, file, indent=4)

pbar.close()
