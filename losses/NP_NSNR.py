import torch.nn as nn
import torch
from matplotlib.pyplot import figure,plot,show,pause,legend

#%% Negative Pearsons Correlation and Negative Signal-to-noise-ratio
# Last modification on 2021/11/14->1-(r+L*SNR) instead of (1-r)+L*(1-SNR)
class NP_NSNR(nn.Module):
    def __init__(self,Lambda,LowF=0.7,upF=3.5,width=0.4):
        super(NP_NSNR,self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Lambda = Lambda
        self.LowF = LowF
        self.upF = upF
        self.width = width
        self.NormaliceK = 1/10.9 #Constant to normalize SNR between -1 and 1
        return
    
    def forward(self, sample:list):
        assert len(sample)>=3, print('=>[NP_NSNR] ERROR, sample must have 3 values [y_hat, y, time]')
        rppg = sample[0]
        gt = sample[1]
        time = sample[2]
        loss = 0        
        for i in range(rppg.shape[0]):
            ##############################
            # PEARSON'S CORRELATION
            ##############################
            sum_x = torch.sum(rppg[i])                # x
            sum_y = torch.sum(gt[i])               # y
            sum_xy = torch.sum(rppg[i]*gt[i])        # xy
            sum_x2 = torch.sum(torch.pow(rppg[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(gt[i],2)) # y^2
            N = rppg.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))   
            ##############################
            # SNR
            ##############################
            N = rppg.shape[-1]*3
            Fs = 1/time[i].diff().mean()
            freq = torch.arange(0,N,1,device=self.device)*Fs/N            
            fft = torch.abs(torch.fft.fft(rppg[i],dim=-1,n=N))**2
            gt_fft = torch.abs(torch.fft.fft(gt[i],dim=-1,n=N))**2
            fft = fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq>self.upF,freq<self.LowF).to(self.device),0)
            PPG_peaksLoc = freq[gt_fft.argmax()]
            mask = torch.zeros(fft.shape[-1],dtype=torch.bool,device=self.device)
            mask = mask.masked_fill(torch.logical_and(freq<PPG_peaksLoc+(self.width/2),PPG_peaksLoc-(self.width/2)<freq).to(self.device),1)#Main signal
            mask = mask.masked_fill(torch.logical_and(freq<PPG_peaksLoc*2+(self.width/2),PPG_peaksLoc*2-(self.width/2)<freq).to(self.device),1)#Armonic
            power = fft*mask
            noise = fft*mask.logical_not().to(self.device)
            SNR = (10*torch.log10(power.sum()/noise.sum()))*self.NormaliceK
            ##############################
            # JOIN BOTH LOSS FUNCTION
            ##############################
            #loss += NegPearson+(self.Lambda*NSNR)
            loss += 1 - (pearson+(self.Lambda*SNR))  
            
        loss = loss/rppg.shape[0]
        return loss  

def stand_alone():
    """ Function to test the loss function
    """
    
    def getFrequencyFromFFT(x,time):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LowF = 0.7
        upF = 3.5
        width = 0.4
        N = x.shape[-1]*3
        Fs = 1/time.diff().mean()
        freq = torch.arange(0,N,1,device=device)*Fs/N            
        fft = torch.abs(torch.fft.fft(x,dim=-1,n=N))**2
        fft = fft.masked_fill(torch.logical_or(freq>upF,freq<LowF).to(device),0)
        return freq[fft.argmax()].detach().to('cpu').item() 
    
    def genSinfromFr(f_sin,sampling_rate,size=128):   
        ## Definir la frecuencia de muestreo, cada cuántos Hz quiero plotear una muestra ejm = 100
        x = np.array([i*sampling_rate for i in range(size)])
        return np.sin(2*np.pi*f_sin*x)

    def plot1SampleTensor(t,x):
        figure()
        plot(t[0].detach().to('cpu').numpy(),x[0].detach().to('cpu').numpy())
        show()

    import numpy as np
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    clean = np.load(r'E:\repos\networks\utils\samples\clean.npy')
    noise = np.load(r'E:\repos\networks\utils\samples\noise.npy')       
    # Create batch
    clean = torch.tensor(np.tile(clean[0:128],(batch_size,1)),device=device,dtype=torch.float32)
    noise = torch.tensor(np.tile(noise[0:128],(batch_size,1)),device=device,dtype=torch.float32)
    # Create time vector
    sr = 0.04#sampling rate    
    timeTrace = np.array([i*sr for i in range(clean.shape[-1])])
    timeTrace = torch.tensor(np.tile(timeTrace,(batch_size,1)),device=device,dtype=torch.float32)
    # Get the frequency of the clean signal by its FFT
    fr = getFrequencyFromFFT(clean[0],timeTrace[0])
    # Create sinusoid with that main frequency
    sinusoid = genSinfromFr(fr,sr)
    sinusoid = torch.tensor(np.tile(sinusoid,(batch_size,1)),device=device,dtype=torch.float32)
    plot1SampleTensor(timeTrace,clean)
    plot1SampleTensor(timeTrace,sinusoid)
    plot1SampleTensor(timeTrace,noise)
    getFrequencyFromFFT(sinusoid[0],timeTrace[0])
    loss = NP_NSNR(Lambda=2)
    val_1 = loss([sinusoid,clean,timeTrace])
    val_2 = loss([noise,clean,timeTrace])
    val_3 = loss([clean,clean,timeTrace])


if __name__ == '__main__':
    pass
    #stand_alone()