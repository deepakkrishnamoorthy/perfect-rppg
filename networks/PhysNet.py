import torch.nn as nn
import torch
from models import MemModule
#%% PHYSNET NETWORK
class PhysNet(nn.Module):
    def __init__(self, frames=128):  
        super(PhysNet, self).__init__()
        
        # Syntax
        # nn.Conv3d(in_channels: int,out_channels: int,kernel_size(depth,height,width), stride, padding)
        # nn.BatchNorm3d(number_features)
        # nn.ReLu(inplace=True)=inplace=True means that it will modify the input directly, without allocating any additional output.
        # nn.ConvTranspose3d(stride,padding,)
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T, 128,128]
        x_visual = x
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [b, 16, 128,128,128]
        x = self.MaxpoolSpa(x)       # x [b, 16, 128,64,64]
        
        x = self.ConvBlock2(x)		    # x [b, 32, 128,64,64]
        x_visual6464 = self.ConvBlock3(x)	    	#x [b, 64, 128,64,64]
        x = self.MaxpoolSpaTem(x_visual6464)      # x [b, 64, 64,32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [b, 64, 64,32,32]
        x_visual3232 = self.ConvBlock5(x)	    	# x [b, 64, 64,32,32]
        x = self.MaxpoolSpaTem(x_visual3232)      # x [b, 64, 32,16,16]
        

        x = self.ConvBlock6(x)		    # x [b, 64, 32,16,16]
        x_visual1616 = self.ConvBlock7(x)	    	# x [b, 64, 32,16,16]
        x = self.MaxpoolSpa(x_visual1616)      # x [b, 64, 32,8,8]

        x = self.ConvBlock8(x)		    # x [b, 64, 32,8,8]
        x = self.ConvBlock9(x)		    # x [b, 64, 32,8,8]
        x = self.upsample(x)		    # x [b, 64, 64,8,8]
        x = self.upsample2(x)		    # x [b, 64, 128,8,8]
        
        
        x = self.poolspa(x)     # x [b, 64, 128,1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [b, 1, 128,1,1] 
        
        rPPG = x.view(-1,length)            
        

        return rPPG, x_visual, x_visual3232, x_visual1616

#%% P64 # PhysNet with input 64
class P64(nn.Module):
    def __init__(self, frames=128):  
        super(P64, self).__init__()
        
        # Syntax
        # nn.Conv3d(in_channels: int,out_channels: int,kernel_size(depth,height,width), stride, padding)
        # nn.BatchNorm3d(number_features)
        # nn.ReLu(inplace=True)=inplace=True means that it will modify the input directly, without allocating any additional output.
        # nn.ConvTranspose3d(stride,padding,)
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x): # x [b, 3, T, 64,64]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x) # [b,3,T=128,64,64] => [b,16,T=128,64,64]
        
        x = self.ConvBlock2(x) # [b,16,T=128,64,64] => [b,32,T=128,64,64]
        x = self.ConvBlock3(x) # [b,32,T=128,64,64] => [b,64,T=128,64,64]
        x = self.MaxpoolSpaTem(x) # [b,64,T=128,64,64]  => [b,64,T=64,32,32]
        
        x = self.ConvBlock4(x)		    # [b,64,T=64,32,32]
        x = self.ConvBlock5(x) # [b,64,T=64,32,32]
        x = self.MaxpoolSpaTem(x) # [b,64,T=64,32,32] => [b,64,T=32,16,16]
        

        x = self.ConvBlock6(x) # [b,64,T=32,16,16]
        x = self.ConvBlock7(x) # [b,64,T=32,16,16]
        x = self.MaxpoolSpa(x) # [b,64,T=32,16,16] => [b,64,T=32,8,8]

        x = self.ConvBlock8(x) # [b,64,T=32,8,8]
        x = self.ConvBlock9(x) # [b,64,T=32,8,8]
        x = self.upsample(x) # [b,64,T=32,8,8] => [b,64,T=64,8,8]
        x = self.upsample2(x) # [b,64,T=64,8,8] => [b,64,T=128,8,8]
        
        
        x = self.poolspa(x) # [b,64,T=128,8,8] => [b,64,T=128,1,1]
        x = self.ConvBlock10(x) # [b,64,T=128,1,1]
        
        rPPG = x.view(-1,length) # [b,128]          
        

        return rPPG, rPPG, rPPG, rPPG
    
#%% PS1
class PS1(nn.Module):
    def __init__(self, frames=128):  
        super(PS1, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
       
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T, 128,128]
        [batch,channel,length,width,height] = x.shape          
        x = self.ConvBlock1(x)		     # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x = self.ConvBlock3(x)	    	# x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)      # x [64, T/4, 16,16]        

        x = self.ConvBlock5(x)		    # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x)      # x [64, T/4, 8,8]

        x = self.ConvBlock6(x)		    # x [64, T/4, 8, 8]

        x = self.upsample(x)		    # x [64, T/2, 8, 8]
        x = self.upsample2(x)		    # x [64, T, 8, 8]
        
        
        x = self.poolspa(x)     # x [64, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length)                    

        return rPPG, rPPG, rPPG, rPPG    

#%% PS2
class PS2(nn.Module):
    def __init__(self, frames=128):  
        super(PS2, self).__init__()
        
        # Syntax
        # nn.Conv3d(in_channels: int,out_channels: int,kernel_size(depth,height,width), stride, padding)
        # nn.BatchNorm3d(number_features)
        # nn.ReLu(inplace=True)=inplace=True means that it will modify the input directly, without allocating any additional output.
        # nn.ConvTranspose3d(stride,padding,)
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [b, 3, T, 128, 128]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [b, 16, T=128, 128, 128]
        x = self.MaxpoolSpa(x)       # x [b, 16, T=128, 64, 64]
        
        x = self.ConvBlock2(x)		    # x [b, 32, T=128, 64, 64]
        x = self.ConvBlock3(x)	    	# x [b, 64, T=128, 64, 64]
        x = self.MaxpoolSpaTem(x)      # x [b, 64, T=64, 32, 32]
        
        x = self.ConvBlock4(x)		    # x [b, 64, T=64, 32, 32]
        x = self.MaxpoolSpaTem(x)      # x [b, 64, T=32, 16, 16]

        x = self.MaxpoolSpa(x)      # x [b, 64, T=32, 8, 8]

        x = self.upsample(x)		    # x [b, 64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [b, 64, T=128, 8, 8]
        
        
        x = self.poolspa(x)     # x [b, 64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [b, 1, T=128, 1, 1]
        
        rPPG = x.view(-1,length)            
        

        return rPPG, rPPG, rPPG, rPPG
    
#%% PS3
class PS3(nn.Module):
    def __init__(self, frames=128):  
        super(PS3, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )        
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [b, 3, T, 128, 128]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [b, 32, T=128, 128, 128]
        x = self.MaxpoolSpa(x)       # x [b, 32, T=128, 64, 64]
        
        x = self.ConvBlock3(x)	    	# x [b, 64, T=128, 64, 64]
        x = self.MaxpoolSpaTem(x)      # x [b, 64, T=64, 32, 32]

        x = self.MaxpoolSpaTem(x)      # x [b, 64, T=32, 16, 16]

        x = self.MaxpoolSpa(x)      # x [b, 64, T=32, 8, 8]

        x = self.upsample(x)		    # x [b, 64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [b, 64, T=128, 8, 8]
        
        
        x = self.poolspa(x)     # x [b, 64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [b, 1, T=128, 1, 1]
        
        rPPG = x.view(-1,length)            
        

        return rPPG, rPPG, rPPG, rPPG

#%% PS4
class PS4(nn.Module):
    def __init__(self, frames=128):  
        super(PS4, self).__init__()


        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 64, [1,6,6],stride=(1,2,2), padding=[0,2,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock1v1 = nn.Sequential(
            nn.Conv3d(3, 64, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
       
        self.MaxpoolSpaTem2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 4, 4), stride=(2, 4, 4))  
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))      
        
    def forward(self, x):	    	# x [b, 3, T, 128, 128]
    
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [b, 64, T=128, 64, 64]
        
        x = self.MaxpoolSpaTem(x)     # x [b, 64, T=64, 16, 16]
        
        x = self.MaxpoolSpaTem2(x)     # x [b, 64, T=32, 8, 8]

        x = self.upsample(x)		    # x [b, 64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [b, 64, T=128, 8, 8]
        
        
        x = self.poolspa(x)     # x [b, 64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [b, 1, T=128, 1, 1]
        
        rPPG = x.view(-1,length)    
        return rPPG, rPPG, rPPG, rPPG      
      
#%% PB1
class PB1(nn.Module):
    def __init__(self, frames=128):  
        super(PB1, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )    
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(128),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(128),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(128, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))       
        
    def forward(self, x):	    	# x [b, 3, T, 128, 128]

        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x = self.ConvBlock3(x)	    	# x [32, T, 64,64]

        x = self.MaxpoolSpaTem(x)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)	    	# x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)      # x [64, T/4, 16,16]
        

        x = self.ConvBlock6(x)		    # x [128, T/4, 16,16]
        x = self.ConvBlock7(x)	    	# x [128, T/4, 16,16]
        x = self.MaxpoolSpa(x)      # x [128, T/4, 8,8]

        x = self.ConvBlock8(x)		    # x [128, T/4, 8, 8]
        x = self.ConvBlock9(x)		    # x [128, T/4, 8, 8]
        
        x = self.upsample(x)		    # x [128, T/2, 8, 8]
        x = self.upsample2(x)		    # x [128, T, 8, 8]
        
        
        x = self.poolspa(x)     # x [128, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length) 
        return rPPG, rPPG, rPPG, rPPG                   
    
#%% PB2
class PB2(nn.Module):
    def __init__(self, frames=128):  
        super(PB2, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock1_5 = nn.Sequential(
            nn.Conv3d(16, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )        
        
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )    
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(128, 128, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(128),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(128),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(128, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))       
        
    def forward(self, x):	    	# x [b, 3, T, 128, 128]

        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [16, T, 128,128]
        x = self.ConvBlock1_5(x)		     # x [16, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x = self.ConvBlock3(x)	    	# x [32, T, 64,64]

        x = self.MaxpoolSpaTem(x)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [128, T/2, 32,32]
        x = self.ConvBlock5(x)	    	# x [128, T/2, 32,32]
        x = self.MaxpoolSpaTem(x)      # x [128, T/4, 16,16]
        

        x = self.ConvBlock6(x)		    # x [128, T/4, 16,16]
        x = self.ConvBlock7(x)	    	# x [128, T/4, 16,16]
        x = self.MaxpoolSpa(x)      # x [128, T/4, 8,8]

        x = self.ConvBlock8(x)		    # x [128, T/4, 8, 8]
        x = self.ConvBlock9(x)		    # x [128, T/4, 8, 8]
        
        x = self.upsample(x)		    # x [128, T/2, 8, 8]
        x = self.upsample2(x)		    # x [128, T, 8, 8]
        
        
        x = self.poolspa(x)     # x [128, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length) 
        return rPPG, rPPG, rPPG, rPPG      

#%% UBS1
class UBS1(nn.Module):
    def __init__(self, frames=128):  
        super(UBS1, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        #self.MaxpoolSpaTem1 = nn.MaxPool3d((4, 4, 4), stride=4)

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 128,128]
        x_visual = x
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 128,128]->[16, T=128, 128, 128]
        x = self.MaxpoolSpaTem1(x)       # x [16, T=128, 128,128]->[16, T=64, 32, 32]

        x = self.ConvBlock2(x)		     # x [16, T=64, 32, 32]->[32, T=64, 32, 32]
        x = self.MaxpoolSpaTem1(x)       # x [32, T=64, 32, 32]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG
    
#%% UB64
class UB64(nn.Module):
    def __init__(self, frames=128):  
        super(UB64, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 64, 64]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 64,64]->[16, T=128,  64,64]
        x = self.MaxpoolSpaTem(x)       # x [16, T=128, 64,64]->[16, T=64, 32, 32]

        x = self.ConvBlock2(x)		     # x [16, T=64, 32, 32]->[32, T=64, 32, 32]
        x = self.MaxpoolSpaTem1(x)       # x [32, T=64, 32, 32]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG    

#%% UB32
class UB32(nn.Module):
    def __init__(self, frames=128):  
        super(UB32, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 32, 32]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 32,32]->[16, T=128,  32,32]
        x = self.MaxpoolTem(x)       # x [16, T=128, 32,32]->[16, T=64, 32, 32]

        x = self.ConvBlock2(x)		     # x [16, T=64, 32, 32]->[32, T=64, 32, 32]
        x = self.MaxpoolSpaTem1(x)       # x [32, T=64, 32, 32]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG 

#%% UB16
class UB16(nn.Module):
    def __init__(self, frames=128):  
        super(UB16, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 16, 16]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 16,16]->[16, T=128,  16,16]
        x = self.MaxpoolTem(x)       # x [16, T=128, 16,16]->[16, T=64, 16, 16]

        x = self.ConvBlock2(x)		     # x [16, T=64, 16, 16]->[32, T=64, 16, 16]
        x = self.MaxpoolSpaTem(x)       # x [32, T=64, 16, 16]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG 

#%% UB8-RGBYUV
class UB8(nn.Module):
    def __init__(self, frames=128):  
        super(UB8, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        #self.gru = nn.GRU(64, 64, batch_first=True)
        
        #self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=feature_num_x2, shrink_thres =shrink_thres)
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((128,1,1))
        
        #self.dropout = nn.Dropout(0.30) #new-line added

        
    def forward(self, x):	    	# x [6, T=128, 8, 8]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [6, T=128, 8, 8]->[16, T=128,  8, 8]
        x = self.MaxpoolTem(x)       # x [16, T=128, 8, 8]->[16, T=64, 8, 8]
        #x = self.dropout(x)         #new-line added

        x = self.ConvBlock2(x)		     # x [16, T=64, 8, 8]->[32, T=64, 8, 8]
        x = self.MaxpoolTem(x)       # x [32, T=64, 8, 8]->[32, T=32, 8, 8]
        #x = self.dropout(x)         #new-line added
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        #x = self.dropout(x)
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG 
    
#net = UB8()
#print(net)


#%% UB8 with LSTM Layers
class UB8_LSTMLAYERS(nn.Module):
    def __init__(self, args, frames=128):  
        super(UB8_LSTMLAYERS, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))
        
        if args.n_LSTM_layers > 0:
            self.uses_LSTM = True
            self.LSTM = nn.LSTM(input_size=1,#features
                                hidden_size=128,
                                num_layers=args.n_LSTM_layers,
                                batch_first=True,
                                dropout=args.LSTM_dropout,
                                bidirectional=False)
        else:
            self.uses_LSTM = False
            
        self.init_weights() # Initilize weights
            
    def init_weights(self):
        print('=>[UB8] Initializating weights')
        for (name_layer,layer) in self.named_children():
            # LSTM WEIGHTS AND BIAS INITIALIZATION
            if name_layer.startswith('LSTM'):
                """
                Use orthogonal init for recurrent layers, xavier uniform for input layers
                Bias is 0 (except for forget gate for first LSTM layer)
                """
                for name_param, param in layer.named_parameters():
                    if 'weight_ih' in name_param:# input
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif 'weight_hh' in name_param:# recurrent
                        torch.nn.init.orthogonal_(param.data,gain=1)
                    elif 'bias' in name_param:# Bias
                        torch.nn.init.zeros_(param.data)
                        if name_layer=='LSTM':
                            param.data[layer.hidden_size:2 * layer.hidden_size] = 1#unit_forget_bias=True
        

        
    def forward(self, x):	    	# x [3, T=128, 8, 8]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 8, 8]->[16, T=128,  8, 8]
        x = self.MaxpoolTem(x)       # x [16, T=128, 8, 8]->[16, T=64, 8, 8]

        x = self.ConvBlock2(x)		     # x [16, T=64, 8, 8]->[32, T=64, 8, 8]
        x = self.MaxpoolTem(x)       # x [32, T=64, 8, 8]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        if self.uses_LSTM == False:
            rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length) 
        else:
            x = self.LSTM(x.view(-1,length,1))
            rPPG = x[0][:,-1,:].view(-1,length)
        
        return rPPG, rPPG, rPPG, rPPG 


#%% UB4
class UB4(nn.Module):
    def __init__(self, frames=128):  
        super(UB4, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 4, 4]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 4, 4]->[16, T=128,  4, 4]
        x = self.MaxpoolTem(x)       # x [16, T=128, 4, 4]->[16, T=64, 4, 4]

        x = self.ConvBlock2(x)		     # x [16, T=64, 4, 4]->[32, T=64, 4, 4]
        x = self.MaxpoolTem(x)       # x [32, T=64, 4, 4]->[32, T=32, 4, 4]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 4, 4]->[64, T=32, 4, 4]
        x = self.ConvBlock4(x)		    # x [64, T=32, 4, 4]->[64, T=32, 4, 4]
        x = self.upsample(x)		    # x [64, T=32, 4, 4]->[64, T=64, 4, 4]
        x = self.upsample2(x)		    # x [64, T=64, 4, 4]->[64, T=128, 4, 4]
        
        x = self.poolspa(x)     # x [64, T=128, 4, 4]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG 
    
#%% UB2
class UB2(nn.Module):
    def __init__(self, frames=128):  
        super(UB2, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 2, 2]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 2, 2]->[16, T=128,  2, 2]
        x = self.MaxpoolTem(x)       # x [16, T=128, 2, 2]->[16, T=64, 2, 2]

        x = self.ConvBlock2(x)		     # x [16, T=64, 2, 2]->[32, T=64, 2, 2]
        x = self.MaxpoolTem(x)       # x [32, T=64, 2, 2]->[32, T=32, 2, 2]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 2, 2]->[64, T=32, 2, 2]
        x = self.ConvBlock4(x)		    # x [64, T=32, 2, 2]->[64, T=32, 2, 2]
        x = self.upsample(x)		    # x [64, T=32, 2, 2]->[64, T=64, 2, 2]
        x = self.upsample2(x)		    # x [64, T=64, 2, 2]->[64, T=128, 2, 2]
        
        x = self.poolspa(x)     # x [64, T=128, 4, 4]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        rPPG = x.view(-1,length) # [128]        rPPG = x.view(-1,length)   
        
        return rPPG, rPPG, rPPG, rPPG     
#%% UB64
class UB64HR(nn.Module):
    def __init__(self, frames=128):  
        super(UB64, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
       
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.HR_nn = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8, out_features=1),
        )    
        
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.MaxpoolSpaTem1 = nn.MaxPool3d((2, 4, 4), stride=(2,4,4))
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T=128, 64, 64]
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T=128, 64,64]->[16, T=128,  64,64]
        x = self.MaxpoolSpaTem(x)       # x [16, T=128, 64,64]->[16, T=64, 32, 32]

        x = self.ConvBlock2(x)		     # x [16, T=64, 32, 32]->[32, T=64, 32, 32]
        x = self.MaxpoolSpaTem1(x)       # x [32, T=64, 32, 32]->[32, T=32, 8, 8]
        
        x = self.ConvBlock3(x)		    # x [32, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.ConvBlock4(x)		    # x [64, T=32, 8, 8]->[64, T=32, 8, 8]
        x = self.upsample(x)		    # x [64, T=32, 8, 8]->[64, T=64, 8, 8]
        x = self.upsample2(x)		    # x [64, T=64, 8, 8]->[64, T=128, 8, 8]
        
        x = self.poolspa(x)     # x [64, T=128, 8, 8]->[64, T=128, 1, 1]
        x = self.ConvBlock10(x)    # x [64, T=128, 1, 1]->[1, T=128, 1, 1]
        
        HR = self.HR_nn(x) # [1, T=128, 1, 1]->[1]
        
        rPPG = x.view(-1,length) # [128]        rPPG ouput  

        
        return rPPG, HR, rPPG, rPPG