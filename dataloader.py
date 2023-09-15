import os
import random
from matplotlib import image as Img
import torch
import json
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from verification import balance_rectification



class image_classif_dataset(Dataset):

    def __init__(self, data_paths, train, command_list):
        self.data_paths = data_paths
        self.train_mode = train
        self.command_lines = []
        for file_path in command_list:
            with open(file_path, 'r') as f:
                self.command_lines.append(f.readlines())
        
    def __getitem__(self, idx):
        image = Img.imread(self.data_paths[idx][2])[120:]   #To keep half of the image
        image = torch.tensor(image)
        image = torch.swapaxes(image, -1, 0)
        image = torch.swapaxes(image, -1, 1)

        command_dict = json.loads(self.command_lines[self.data_paths[idx][0]][self.data_paths[idx][1]])

        if command_dict['steering_angle'] > 0.54:
            direction = 0
        elif command_dict['steering_angle'] > 0.38:
            direction = 1
        elif command_dict['steering_angle'] > 0.23:
            direction = 2
        elif command_dict['steering_angle'] > 0.07:
            direction = 3
        elif command_dict['steering_angle'] > -0.07:
            direction = 4
        elif command_dict['steering_angle'] > -0.23:
            direction = 5
        elif command_dict['steering_angle'] > -0.38:
            direction = 6
        elif command_dict['steering_angle'] > -0.54:
            direction = 7
        else:
            direction = 8

        direction = torch.tensor(direction)
        speed = torch.tensor(command_dict['speed'])
        
        return image.float(), direction, speed.float()
    
    def __len__(self):
        return len(self.data_paths)




def get_test_dataloader(Folder,n,seed):
    data_list = []
    command_list = []
    val=[]
    test=[]
    val_dataset=[]
    test_dataset=[]
    train_dataloader=[]
    for j in range (n):
        command_list=[]
        data_list=[]
        for d,data_file in enumerate(Folder[j]):
            command_list.append(os.path.join(data_file, 'commands.json'))
            command_line=[]
            for file_path in command_list:
                with open(file_path, 'r') as f:
                    command_line.append(f.readlines())
                    
            for instant in range (len(os.listdir(os.path.join(data_file,'images')))):
                topics_list = []
                topics_list.append(d)
                topics_list.append(instant)
                com=json.loads(command_line[0][instant])["img_path"]
                topics_list.append(os.path.join(data_file, os.path.join('images', f'{com}.jpeg')))
                data_list.append(topics_list)
    
        random.Random(seed).shuffle(data_list)
        train_dataset= image_classif_dataset(data_list[:int(0.8*len(data_list))], True, command_list)
        val.append(image_classif_dataset(data_list[int(0.8*len(data_list)):int(0.9*len(data_list))], False, command_list))
        test.append(image_classif_dataset(data_list[int(0.9*len(data_list)):], False, command_list))
        train_dataloader.append(DataLoader(train_dataset, batch_size=16, shuffle=True))
        
    for i in range(len(val)):
        val_dataset= ConcatDataset([ val_dataset, val[i]])
        test_dataset= ConcatDataset([ test_dataset, test[i]])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    return train_dataloader, val_loader, test_loader


def get_balanced_dataloader(Folder):

    data_list = []
    command_list = []
    multi=1
    for d,data_file in enumerate(Folder):
        command_list.append(os.path.join(data_file, 'commands.json'))
        command_line=[]
        for file_path in command_list:
                with open(file_path, 'r') as f:
                  command_line.append(f.readlines())
        # to balance the data 
        prop, _=balance_rectification(bins=9,folder=Folder,taux=1)
        for instant in range (len(os.listdir(os.path.join(data_file,'images')))):
            Steer=json.loads(command_line[0][instant])["steering_angle"]
            if Steer > 0.54:
                multi= prop[8]
            elif Steer > 0.38:
                multi=prop[7]
            elif Steer > 0.23:
                multi=prop[6]
            elif Steer > 0.07:
                multi=prop[5]
            elif Steer > -0.07:
                multi=prop[4]
            elif Steer > -0.23:
                multi=prop[3]
            elif Steer > -0.38:
                multi=prop[2]
            elif Steer > -0.54:
                multi=prop[1]
            else:
                multi=prop[0]
            
            # increase the number of data that are least represented overall
            for k in range (multi):
                topics_list = []
                topics_list.append(d)
                topics_list.append(instant)
                com=json.loads(command_line[0][instant])["img_path"]
                topics_list.append(os.path.join(data_file, os.path.join('images', f'{com}')))
                data_list.append(topics_list)
                
    random.Random(3).shuffle(data_list)
    data_list=data_list[:int(0.5*len(data_list))]
    
    
    train_dataloader = DataLoader(image_classif_dataset(data_list[:int(0.8*len(data_list))], True, command_list), batch_size=16, shuffle=True)
    val_dataloader = DataLoader(image_classif_dataset(data_list[int(0.8*len(data_list)):int(0.9*len(data_list))], False, command_list), batch_size= 16,shuffle=True)
    test_dataloader= DataLoader(image_classif_dataset(data_list[int(0.9*len(data_list)):], False, command_list), batch_size= 16,shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader