import torch.nn as nn
import torch

#%% Negative Pearsons correlation as loss function
# call Negative Pearsons correlation as loss function
class NP(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(NP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def forward(self, sample:list):
        # assert len(sample)==2, print('=>[NP] ERROR, sample must have 2 values [y_hat , y]')
        preds = sample[0]
        labels =  sample[1]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x,RG=True
            sum_y = torch.sum(labels[i])               # y,RG=False
            sum_xy = torch.sum(preds[i]*labels[i])        # xy,,RG=True
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2,RG=True
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2,RG=False
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))#,RG=True
            
            loss += 1 - (pearson)
            
            
        loss = loss/preds.shape[0]
        return loss
