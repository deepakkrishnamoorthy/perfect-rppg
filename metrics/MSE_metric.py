import torch.nn as nn
import torch
from matplotlib.pyplot import figure,plot,show,legend,title
import numpy as np

#%% MSE : I just duplicate torch MSE with the code way I am using in loss functions 
class MSE_metric(nn.Module):
    def __init__(self,Lambda=1):
        super(MSE_metric,self).__init__()
        self.has_to_be_minimized = True
        return
   
    def forward(self, preds, labels, t=None):       # tensor [Batch, Temporal]
        MSE = torch.nn.MSELoss(reduction='mean')
        return MSE(preds,labels)

def stand_alone(Run=False):
    """ Function to test the metric
    """
    def plot1SampleTensor(x):
        figure()
        plot(x[0].detach().to('cpu').numpy())
        show()
        
    if Run:   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 4
        clean = np.load(r'E:\repos\networks\utils\samples\clean.npy')
        noise = np.load(r'E:\repos\networks\utils\samples\noise.npy')
        # Create batch
        clean = torch.tensor(np.tile(clean[0:128],(batch_size,1)),device=device,dtype=torch.float32)
        noise = torch.tensor(np.tile(noise[0:128],(batch_size,1)),device=device,dtype=torch.float32)
        # plot
        plot1SampleTensor(clean)
        plot1SampleTensor(noise)
        # Measure    
        metric = MSE_metric(Lambda=1)  
        print(metric(clean,noise).item())
        print(metric(clean,clean).item())
        print(metric(noise,noise).item())

if __name__ == '__main__':
    stand_alone()
