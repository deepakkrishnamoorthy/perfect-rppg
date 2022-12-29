import torch.nn as nn
import torch
from matplotlib.pyplot import figure,plot,show,legend,title

#%% Negative Pearsons correlation as loss function
# call Negative Pearsons correlation as loss function
class r_metric(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self,Lambda=1):
        super(r_metric,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_to_be_minimized = False
        
        return
    def forward(self, preds, labels, t=None):       # tensor [Batch, Temporal]
        r = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x,RG=True
            sum_y = torch.sum(labels[i])               # y,RG=False
            sum_xy = torch.sum(preds[i]*labels[i])        # xy,,RG=True
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2,RG=True
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2,RG=False
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))#,RG=True
            r += pearson

        r = r/preds.shape[0]
        return r

def stand_alone(Run=False):
    """ Function to test the metric
    """
    def plot1SampleTensor(x):
        figure()
        plot(x[0].detach().to('cpu').numpy())
        show()
    if Run:
        import numpy as np
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 4
        clean = np.load(r'E:\repos\deep_rppg\utils\samples\clean.npy')
        noise = np.load(r'E:\repos\deep_rppg\utils\samples\noise.npy')
        # Create batch
        clean = torch.tensor(np.tile(clean[0:128],(batch_size,1)),device=device,dtype=torch.float32)
        noise = torch.tensor(np.tile(noise[0:128],(batch_size,1)),device=device,dtype=torch.float32)
        # plot
        plot1SampleTensor(clean)
        plot1SampleTensor(noise)
        # Measurement
        loss = r_metric(Lambda=1)  
        print(loss(clean,noise).item())
        print(loss(clean,clean).item())
        print(loss(noise,noise).item())


if __name__ == '__main__':
    stand_alone()
