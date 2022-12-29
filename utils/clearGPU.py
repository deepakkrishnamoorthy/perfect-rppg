import torch
import gc
#%%

def clearGPU(VERBOSE:bool=False):
    if VERBOSE>0: print('=>[clearGPU] Done.')
    gc.collect()
    torch.cuda.empty_cache()  
