import os
import random
from configuration import cfg
import matplotlib.pyplot as plt
from matplotlib import image as Img
import torch
import numpy as np
import torchvision
import torcheval
import json
import tqdm
from torch.utils.data import Dataset
import statistics
import torch.nn as nn
from model import get_mobilenet_classif_1
from torcheval.metrics import MulticlassAccuracy
from Outils import RMSE, radian_9
from dataloader import get_test_dataloader
import torch.optim as optim
from aggregation import Avg_server_aggregation, Proj_server_aggregation, Best_losses_proj, Best_losses_avg



##hyper-parameters for centralized federated learning
num_clients = 6  
num_selected = 6
num_rounds = 50
epochs = 5
contributors = 2

criterion = nn.CrossEntropyLoss()
metric_steer = MulticlassAccuracy()
mesure = RMSE()


def client_update(client_model, optimizer, train_loader, epoch):
    torch.manual_seed(0)
    train_list_loss=[]
    client_model= nn.DataParallel(client_model)
    client_model.to(device)

    model.train()
    for e in range (epoch):
        train_range = tqdm.tqdm(train_loader)
        for i, (data, direction, speed) in enumerate(train_range):
            
            if torch.cuda.is_available():
                data=data.cuda()
                direction=direction.cuda()
                speed=speed.cuda()
                
            optimizer.zero_grad()
            
            output=client_model(data)
            _, pred= torch.max(output.data, 1)
            loss = criterion(output, direction)
            train_list_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            train_range.set_description("TRAIN -> epoch: %4d " % (e))
            train_range.refresh()
    return loss.item()
            



def test(global_model, test_loader):
    model.eval()
    test_loss=0
    correct=0

    mesure_test=[]
    with torch.no_grad():
        for j, (data, direction, speed) in enumerate(test_loader):
            if torch.cuda.is_available():
                data=data.cuda()
                direction=direction.cuda()
                speed= speed.cuda()
                
            output =global_model(data)
            test_loss+=criterion(output,direction)
            _, pred = torch.max(output.data, 1)
            correct+=((pred == direction).sum().item())
            pred=radian_9(pred)
            true=radian_9(direction)

            mesure_test.append(mesure(pred,true))
            
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    rmse= statistics.mean(mesure_test)
    return test_loss, acc, rmse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialization of locals and global models
model=get_mobilenet_classif_2()
global_model= get_mobilenet_classif_2()
if torch.cuda.is_available():
    model=model.cuda()
    global_model=global_model.cuda()

client_models= [model for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())
    
opt=[torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]


data_folder=[cfg.DATASET.USED_DATA_FOLDER_1,cfg.DATASET.USED_DATA_FOLDER_4,cfg.DATASET.USED_DATA_FOLDER_5, cfg.DATASET.USED_DATA_FOLDER_6,cfg.DATASET.USED_DATA_FOLDER_8,cfg.DATASET.USED_DATA_FOLDER_9]
train_loader, val_loader, test_loader = get_test_dataloader(data_folder,len(data_folder),3)


checkpoint_count=0
with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'results.txt'), 'a+') as result_file:
    result_file.write("FL : num_selected = 6 / epochs=5 / rounds=50 \n")
    
    
accuracy=[]
mesure_rmse=[]
for r in range(1,num_rounds+1):
    loss=0
    losses_train=[]
    for i in range(num_selected):

        losses=client_update(client_models[i],opt[i],train_loader[i],epoch=epochs)
        loss+=losses
        losses_train.append(losses)
    
    Best_losses_avg(global_model, client_models, losses_train, contributors)
    
    val_loss, val_acc , val_rmse = test(global_model, val_loader)
    accuracy.append(val_acc)
    mesure_rmse.append(val_rmse)
    
    test_loss, test_acc , test_rmse = test(global_model, test_loader)
    
    print('%d-th round' %r)
    print('Validation : average train loss : %0.3g | test loss : %0.3g | test acc : %0.1f | rmse : %0.4f' % (loss / num_selected, val_loss, val_acc*100, val_rmse))
    print('Test : average train loss : %0.3g | test loss : %0.3g | test acc : %0.1f | rmse : %0.4f' % (loss / num_selected, test_loss, test_acc*100, test_rmse))
    checkpoint_count += 1
    print(" ")
    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'results.txt'), 'a+') as result_file:
        result_file.write(f"Val: checkpoint_{checkpoint_count} : train_loss = {loss / num_selected} , val_loss ={val_loss}, val_accuracy = {val_acc}, val_rmse = {val_rmse} \n")
        result_file.write(f"Test: checkpoint_{checkpoint_count}: train_loss = {loss / num_selected} , test_loss ={test_loss}, test_accuracy = {test_acc}, test_rmse = {test_rmse} \n")
        result_file.write(f"\n")

print('Maximum accuracy : %0.1f | Maximum rmse : %0.4f' % (max(accuracy)*100 ,min(mesure_rmse)))
print("rounds : ",np.argmax(accuracy)+1, " et ",np.argmin(mesure_rmse)+1)
