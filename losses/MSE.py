import torch.nn as nn
import torch

#%% Mean Squared Error loss
class MSE_lstmdf(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(MSE_lstmdf, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.MSEloss = torch.nn.MSELoss(reduction='mean')

    
    def forward(self, sample:list):
        # assert len(sample)==2, print('=>[NP] ERROR, sample must have 2 values [y_hat , y]')
        y_hat = sample[0].squeeze()
        y =  sample[1].squeeze()       

        return self.MSEloss(y_hat,y)

