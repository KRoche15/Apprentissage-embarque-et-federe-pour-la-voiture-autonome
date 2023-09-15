import torch 
import numpy as np
from utils import best_index


def Avg_server_aggregation(global_model, client_models):
    W = global_model.state_dict()
    C=[]
    for i in range (len(client_models)):
        C.append(client_models[i].state_dict())
        for k in C[i].keys():
            C[i][k] = C[i][k].float() - W[k].float()
    for k in W.keys():
        W[k]= W[k]+ torch.stack([C[i][k].float() for i in range(len(C))], 0).mean(0)

    global_model.load_state_dict(W)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


def Proj_server_aggregation(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        values = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))])
        shape = values[0].shape 
        for i in range(len(client_models)):
            for j in range(i + 1, len(client_models)):
                g_i = values[i].flatten()
                g_j = values[j].flatten()
                g_i_g_j = torch.dot(g_i, g_j)
                #projection with the Gram-Schmidt algorithm
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (torch.norm(g_j)**2)
                    values[i] = g_i
        values = values.view(len(client_models), *shape)
        global_dict[k] = values.mean(dim=0)
    
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
        

def Best_losses_proj(global_model, client_models,losses_train,n=3):
    best_losses = best_index(losses_train,n)
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        values = torch.stack([client_models[i].state_dict()[k].float() for i in best_losses])
        shape = values[0].shape 
        for i in range(n):
            for j in range(i + 1, n):
                g_i = values[i].flatten()
                g_j = values[j].flatten()
                g_i_g_j = torch.dot(g_i, g_j)
                
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (torch.norm(g_j)**2)
                    values[i] = g_i
        
        values = values.view(n, *shape)
        global_dict[k] = values.mean(dim=0)
    
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def Best_losses_avg(global_model, client_models,losses_train,n):
    best_losses = best_index(losses_train,n)
    W = global_model.state_dict()
    C=[]
    for j in best_losses:
        C.append(client_models[j].state_dict())
    for i in range (len(C)):
        for k in C[i].keys():
            C[i][k] = C[i][k].float() - W[k].float()
    for k in W.keys():
        W[k]= W[k]+ torch.stack([C[i][k].float() for i in range(len(C))], 0).mean(0)

    global_model.load_state_dict(W)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())