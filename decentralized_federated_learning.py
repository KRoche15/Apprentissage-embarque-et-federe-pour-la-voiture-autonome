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
import torch.nn.functional as F
from model import get_mobilenet_classif_2
from torcheval.metrics import MulticlassAccuracy
from Outils import RMSE, radian_9
from dataloader import get_test_dataloader
import torch.optim as optim
from aggregation import Avg_server_aggregation, Proj_server_aggregation, Best_losses_proj, Best_losses_avg



##hyper-parameters for decentralized federated learning
num_clients = 4
num_selected = np.array([2,3,4,5])  #3 selected D1-> D4
seed = 4
num_rounds = 50
epochs = 5

criterion = torch.nn.CrossEntropyLoss()
metric_steer = MulticlassAccuracy()
mesure = RMSE()


def client_update(client_model, optimizer, train_loader, epoch):
    torch.manual_seed(0)
    train_list_loss=[]
    model.train()
    for e in range (epoch):
        train_range = tqdm.tqdm(train_loader)
        for i, (data, direction, speed) in enumerate(train_range):
            
            if torch.cuda.is_available():
                data=data.cuda()
                direction=direction.cuda()
                speed=speed.cuda()
                
            optimizer.zero_grad()
            
            output1=client_model(data)
            _, pred= torch.max(output1.data, 1)
            loss = criterion(output1, direction)
            train_list_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            train_range.set_description("TRAIN -> epoch: %4d " % (e))
            train_range.refresh()
    return loss.item()
            


def test(model, test_loader):
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
                
            output1 =model(data)
            
            test_loss+=criterion(output1,direction)
            _, pred = torch.max(output1.data, 1)
            correct+=((pred == direction).sum().item())
            pred=radian_9(pred)
            true=radian_9(direction)

            mesure_test.append(mesure(pred,true))
            
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    rmse= statistics.mean(mesure_test)
    return test_loss, acc, rmse


#initialization of locals and global models
model=get_mobilenet_classif_2()
global_model= get_mobilenet_classif_2()
if torch.cuda.is_available():
    model=model.cuda()
    global_model=global_model.cuda()

client_models= [model for _ in range(6)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())

opt=[torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]

data_folder=[cfg.DATASET.USED_DATA_FOLDER_1,cfg.DATASET.USED_DATA_FOLDER_4,cfg.DATASET.USED_DATA_FOLDER_5, cfg.DATASET.USED_DATA_FOLDER_6,cfg.DATASET.USED_DATA_FOLDER_8,cfg.DATASET.USED_DATA_FOLDER_9]
train_loader, val_loader, test_loader = get_test_dataloader(data_folder,len(data_folder),seed)


checkpoint_count=0
with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'decentralized_results.txt'), 'a+') as result_file:
    result_file.write("DFL : num_selected = 6 / epoch = 5 / round = 50 \n")
    
accuracy=[]
mesure_rmse=[]
accuracy_test=[]
mesure_rmse_test=[]
Acc=[]
Mes=[]
loss_val=[]
loss_test=[]
for r in range(1,num_rounds+1):
    loss=0
    losses_train=[]
    for i in num_selected :
        losses=client_update(client_models[i],opt[i],train_loader[i],epoch=epochs)
        loss+=losses
        losses_train.append(losses)

    com=np.random.random_integers(0,3,2)
    while com[0]==com[1]:
        com=np.random.random_integers(0,3,2)
    global_client_models= [client_models[k] for k in com]
    Avg_server_aggregation(global_model, global_client_models)

    for j in range (len(num_selected)):
        val_loss, val_acc , val_rmse = test(client_models[j], val_loader)
        loss_val.append(val_loss)
        accuracy.append(val_acc)
        mesure_rmse.append(val_rmse)
        
        test_loss, test_acc , test_rmse = test(client_models[j], test_loader)
        loss_test.append(test_loss)
        accuracy_test.append(test_acc)
        mesure_rmse_test.append(test_rmse)
        
    final_accuracy=sum(accuracy) / len(num_selected)
    final_mesure_rmse=sum(mesure_rmse) / len(mesure_rmse)
    final_accuracy_test=sum(accuracy_test) / len(num_selected)
    final_mesure_rmse_test=sum(mesure_rmse_test) / len(mesure_rmse_test)

    Acc.append(final_accuracy)
    Mes.append(final_mesure_rmse)

    print('%d-th round' %r)
    print('communication between clients:', com[0]+1, " and ", com[1]+1)
    print('Validation : average train loss : %0.3g | test loss : %0.3g | test acc : %0.3f | rmse : %0.4f' % (loss / len(num_selected), sum(loss_val) / len(num_selected), final_accuracy, final_mesure_rmse))
    print('Test : average train loss : %0.3g | test loss : %0.3g | test acc : %0.3f | rmse : %0.4f' % (loss / len(num_selected), sum(loss_test) / len(num_selected), final_accuracy_test, final_mesure_rmse_test))
    # save the model
    torch.save(model.state_dict(), os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH,'ckpt_' + str(checkpoint_count)) + ".ckpt")
    checkpoint_count += 1
    print(" ")
    with open(os.path.join(cfg.TRAIN.CHECKPOINT_SAVE_PATH, 'decentralized_results.txt'), 'a+') as result_file:
        result_file.write(f"Val: checkpoint_{checkpoint_count} : train_loss = {loss / len(num_selected)} , val_loss ={val_loss}, val_accuracy = {final_accuracy}, val_rmse = {final_mesure_rmse} \n")
        result_file.write(f"Test: checkpoint_{checkpoint_count}: train_loss = {loss / len(num_selected)} , test_loss ={test_loss}, test_accuracy = {final_accuracy_test}, test_rmse = {final_mesure_rmse_test} \n")
        result_file.write(f"Communication: {com} , round: {r} \n")
        result_file.write(f"\n")
    
    accuracy=[]
    mesure_rmse=[]
    accuracy_test=[]
    mesure_rmse_test=[]
    loss_val=[]
    loss_test=[]
print('Maximum accuracy : %0.1f | Maximum rmse : %0.4f' % (max(Acc)*100 ,min(Mes)))
print("rounds : ",np.argmax(Acc)+1, " / ",np.argmin(Mes)+1)