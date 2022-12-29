import enum

"""
ENUM CLASES DEFINING THE PARAMETERS IN THE NETWORK
"""

class NetworkModel(enum.Enum):
    PHYSNET = enum.auto()
    RPPGNET = enum.auto()
    P64 = enum.auto() # PhysNet input 65
    PS1 = enum.auto()
    PS2 = enum.auto()
    PS3 = enum.auto()
    PS4 = enum.auto()
    PB1 = enum.auto()
    PB2 = enum.auto()
    UBS1 = enum.auto()
    UB64 = enum.auto() #UBS1 input 64
    UB32 = enum.auto() #UB64 input 32
    UB16 = enum.auto() #UB32 input 16
    UB8 = enum.auto() #UB16 input 8
    RTRPPG = enum.auto() #Is UB8 but with different name
    UB8_LSTMLAYERS = enum.auto() #UB8 with 6 channels and LSTM layers
    UB4 = enum.auto() #UB8 input 4
    UB2 = enum.auto() #UB4 input 2 -- literally same UB4   
    UB64HR = enum.auto()
    UBS2 = enum.auto()
    UBS3 = enum.auto()
    LSTMDF125 = enum.auto()
    LSTMDFMTM128 = enum.auto()
    LSTMDFMTO128 = enum.auto()
    RTRPPG_LSTMDFMTM128 = enum.auto()
    RTRPPG_LSTMDFMTO128 = enum.auto()
   
class DatasetName(enum.Enum):
    MMSE = enum.auto()
    COHFACE = enum.auto()
    VIPL = enum.auto() 
    ECGF = enum.auto()
    UBFC = enum.auto()
    BIGECGF = enum.auto()
    GARAGE_NIR = enum.auto()
    MRL = enum.auto()
    BIGUBFC = enum.auto()
   
class Loss(enum.Enum):
    NP = enum.auto()
    MAEINHR = enum.auto()
    MSEINHR = enum.auto()
    MSE = enum.auto()
    PSNR = enum.auto()
    MSEN1FFT = enum.auto()
    MSEN2FFT = enum.auto()
    NSNR = enum.auto()
    HRMAE = enum.auto()
    NP_MSEN1FFT = enum.auto()
    NP_NSNR = enum.auto()

class Metric(enum.Enum):
    #It will always run hrmae_hrr_snr. But you choose which one use to avoid overfitting
    hrhatmae = enum.auto() # Heart rate Mean Absolute Error measured with HR predicted by the network
    hrmae = enum.auto() # Heart rate Mean Absolute Error measured in rPPG prediction
    hrr = enum.auto()
    snr = enum.auto()

class Optimizer(enum.Enum):
    ADAM = enum.auto()
    ADADELTA = enum.auto()
    
class ColorChannel(enum.Enum):
    RGB = enum.auto()
    YUV = enum.auto() 
    HSV = enum.auto()
    Lab = enum.auto()
    Luv = enum.auto()
    YCrCb = enum.auto()
    NIR = enum.auto()
    