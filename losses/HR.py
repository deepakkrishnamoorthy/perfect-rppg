import torch.nn as nn
import torch

# LOSS TO MEASURE MSE IN HR values
# MSE : I just duplicate torch MSE with the code way I am using in loss functions 
class MSEINHR(nn.Module):
    def __init__(self,Lambda=1,):
        super(MSEINHR,self).__init__()
        self.Lambda = Lambda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LowF = 0.7
        self.upF = 3.5
        self.MSE = torch.nn.MSELoss(reduction='mean')        
        return
 
    def getHRfromPPG(self,ppg,time):
        HR_gt = torch.zeros(ppg.shape[0],dtype=torch.float32,device=self.device)
        N = ppg.shape[-1]*3         
        for i in range(ppg.shape[0]):
            # Find sampling frequency
            Fs = 1/time[i].diff().mean()
            # Create frequency tensor in the FFT and FFT
            freq = torch.arange(0,N,1,device=self.device)*Fs/N# RG=False            
            gt_fft = torch.abs(torch.fft.fft(ppg[i],dim=-1,n=N))**2# RG=False       
            # Remove out-of-range frequencies
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)# RG=False
            # Find HR in GT (we can use argmax)
            PPG_peaksLoc = freq[gt_fft.argmax()]# RG=False
            HR_gt[i] = PPG_peaksLoc
            
        return HR_gt
 
    def forward(self, sample:list):
        assert len(sample)==4, print('=>[MSEINHR] ERROR, sample must have 4 values [y_hat, y, time, HR]')

        gt = sample[1]
        time = sample[2]  
        HR = sample[3].squeeze()
        GTHR = self.getHRfromPPG(gt,time)

        return self.MSE(HR,GTHR)*self.Lambda 
    
# LOSS TO MEASURE MSE IN HR values
# MAE : I just duplicate torch MAE with the code way I am using in loss functions 
class MAEINHR(nn.Module):
    def __init__(self,Lambda=1,):
        super(MAEINHR,self).__init__()
        self.Lambda = Lambda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LowF = 0.7
        self.upF = 3.5
        self.MAE = torch.nn.L1Loss(reduction='mean')        
        return
 
    def getHRfromPPG(self,ppg,time):
        HR_gt = torch.zeros(ppg.shape[0],dtype=torch.float32,device=self.device)
        N = ppg.shape[-1]*3         
        for i in range(ppg.shape[0]):
            # Find sampling frequency
            Fs = 1/time[i].diff().mean()
            # Create frequency tensor in the FFT and FFT
            freq = torch.arange(0,N,1,device=self.device)*Fs/N# RG=False            
            gt_fft = torch.abs(torch.fft.fft(ppg[i],dim=-1,n=N))**2# RG=False       
            # Remove out-of-range frequencies
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)# RG=False
            # Find HR in GT (we can use argmax)
            PPG_peaksLoc = freq[gt_fft.argmax()]# RG=False
            HR_gt[i] = PPG_peaksLoc
            
        return HR_gt
 
    def forward(self, sample:list):
        assert len(sample)>3, print('=>[MSEINHR] ERROR, sample must have 4 values [y_hat, y, time, HR]')

        gt = sample[1]
        time = sample[2]  
        HR = sample[3].squeeze()
        GTHR = self.getHRfromPPG(gt,time)
        
        # If is UB64HR I want to return the vector result instead of the average
        if len(sample)>4:
            if sample[4]['is_eval']:
                return torch.abs(HR-GTHR)

        return self.MAE(HR,GTHR)*self.Lambda  