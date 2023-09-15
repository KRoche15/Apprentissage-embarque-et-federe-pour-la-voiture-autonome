import torch
import os
import json
import numpy as np

def balance_correction(bins=9, folder=[r"C:\Users\...\robot_3"] ,taux=1):
    command_list=[]
    data_list = []
    for d,data_file in enumerate(folder):
            command_list.append(os.path.join(data_file, 'commands.json'))
            command_line=[]
            for file_path in command_list:
                    with open(file_path, 'r') as f:
                        command_line.append(f.readlines())
            for instant in range(len(os.listdir(os.path.join(data_file,'images')))):
                topics_list = []
                topics_list.append(d)
                topics_list.append(instant)
                com=json.loads(command_line[0][instant])["img_path"]
                topics_list.append(os.path.join(data_file, os.path.join('images', f'{com}.jpeg')))
                data_list.append(topics_list)
            
    data_paths=data_list[:int(taux*len(data_list))]
    command_lines = []
    for file_path in command_list:
        with open(file_path, 'r') as f:
            command_lines.append(f.readlines())

    direction=[]
    for idx in range (int(taux*len(data_list))):
        command_dict = json.loads(command_lines[data_paths[idx][0]][data_paths[idx][1]])
        direction.append(list(command_dict.values())[2])
    if bins==9:
        inter=[0.69,0.54,0.38,0.23,0.07,-0.07,-0.23,-0.38,-0.54,-0.69]
    elif bins==15:
        inter=[0.69,0.6,0.51,0.42,0.31,0.23,0.14,0.04,-0.04,-0.14,-0.23,-0.31,-0.42,-0.51,-0.6,-0.69]
    
    hist, bin_edges = np.histogram(direction, bins=bins,range=(-0.69,0.69))
    prop=hist/(len(direction))*100
    return np.int_(np.floor(np.max(prop)/prop)), bin_edges