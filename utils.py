import torch 
import os
import numpy as np

class MSEcustom(torch.nn.Module):
    def __init__(self):
        super(MSEcustom, self).__init__()

    def forward(self, y_true, y_pred):
        result = torch.sum((y_true - y_pred)**2)/y_true.shape[0]
        return result


class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_true, y_pred):
        result=0
        for i in range(len(y_true)):
            result += (y_true[i] - y_pred[i])**2
        result = np.sqrt(result/len(y_true))
        return result
    
    
def radian_9(pred):
    mil=[0.615,0.46,0.305,0.15,0.0,-0.15,-0.305,-0.46,-0.615]
    radian=[]
    for j in range (len(pred)):
        for i in range (9):
            if pred[j]==i:
                radian.append(mil[i])
    return radian

def best_index(vect,n):
    i=0
    index=[]
    while (i<n):
        index.append(np.argmin(vect))
        vect.pop(index[i])
        i+=1
    return index