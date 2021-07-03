f = open("deleted_files.txt", 'r')
import shutil
import time


# genius mode 3k 
FILE_NAMES = []

for line in f:
    
    move_dir = "/Users/anish/Documents/Machine Learning Env/AnishMachineLearning/" + line[line.index(" ") + 1 : ]
    reversed_line = line[::-1]
    reversed_file_name = reversed_line[:reversed_line.index('/')]
    file_name = reversed_file_name[::-1]
    file_name = file_name[:-1]
    
    FILE_NAMES.append(file_name)

import os

for behavior in ['armflapping', 'control']: 
    for video_name in os.listdir('/Users/anish/Documents/Machine Learning Env/AnishMachineLearning/behavior_data/training/' + behavior): 
        if video_name in FILE_NAMES: 
            try: 
                shutil.move('/Users/anish/Documents/Machine Learning Env/AnishMachineLearning/behavior_data/training/' + behavior + '/' + video_name, '/Users/anish/Documents/Machine Learning Env/AnishMachineLearning/behavior_data/' + behavior + '/') 
            except Exception as e:
                print(f"failed on {video_name}")