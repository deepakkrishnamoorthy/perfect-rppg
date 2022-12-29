'''
Script to measure the inference time of all models.
Based on https://deci.ai/resources/blog/measure-inference-time-deep-neural-networks/
'''
from networks.PhysNet import PhysNet, P64, PS1, PS2, PS3, PS4, PB1, PB2, UBS1, UB64, UB32, UB16, UB8, UB4, UB2
from networks.SOA import rPPGNet
from networks.LSTMDF import LSTMDFMTM125, LSTMDFMTM128
import torch
import numpy as np
from utils.clearGPU import clearGPU
import pandas as pd
import time
from time import perf_counter
import torch.nn as nn


def getInferenceTime(modelname:str, devicename:str='auto'):
    if devicename == 'auto':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = devicename
    
    if modelname == 'PHYSNET':
        model = PhysNet().to(device)
    elif modelname == 'rPPGNet':
        model = rPPGNet().to(device)
    elif modelname == 'LSTMDF125':
        model = LSTMDFMTM125(device=devicename).to(device) 
        model.reset_hidden()
    elif modelname == 'LSTMDF128':
        model = LSTMDFMTM128(device=devicename).to(device) 
        model.reset_hidden()
    elif modelname == 'P64':
        model = P64().to(device)               
    elif modelname == 'PS1':
        model = PS1().to(device)
    elif modelname == 'PS2':
        model = PS2().to(device)
    elif modelname == 'PS3':
        model = PS3().to(device)
    elif modelname == 'PB1':
        model = PB1().to(device)
    elif modelname == 'PB2':
        model = PB2().to(device) 
    elif modelname == 'UBS1':
        model = UBS1().to(device) 
    elif modelname == 'UB64':
        model = UB64().to(device) 
    elif modelname == 'UB32':
        model = UB32().to(device) 
    elif modelname == 'UB16':
        model = UB16().to(device)         
    elif modelname == 'UB8':
        model = UB8().to(device)  
    elif modelname == 'UB4':
        model = UB4().to(device)
    elif modelname == 'UB2':
        model = UB2().to(device)        
        
    # Measure total parameters in model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if modelname in ['P64','UB64']:
        dummy_input = torch.randn(1, 3, 128, 64, 64, dtype = torch.float32).to(device)
    elif modelname in ['UB32']:
        dummy_input = torch.randn(1, 3, 128, 32, 32, dtype = torch.float32).to(device)   
    elif modelname in ['UB16']:
        dummy_input = torch.randn(1, 3, 128, 16, 16, dtype = torch.float32).to(device)   
    elif modelname in ['UB8']:
        dummy_input = torch.randn(1, 3, 128, 8, 8, dtype = torch.float32).to(device) 
    elif modelname in ['UB4']:
        dummy_input = torch.randn(1, 3, 128, 4, 4, dtype = torch.float32).to(device)
    elif modelname in ['UB2']:
        dummy_input = torch.randn(1, 3, 128, 2, 2, dtype = torch.float32).to(device)        
    elif modelname in ['rPPGNet']:
        dummy_input = torch.randn(1, 3, 64, 128, 128, dtype = torch.float32).to(device)
    elif modelname in ['LSTMDF125']:
        dummy_input = torch.randn(1, 125, 1, dtype = torch.float32).to(device)        
    elif modelname in ['LSTMDF128']:
        dummy_input = torch.randn(1, 128, 1, dtype = torch.float32).to(device)   
    else:    
        dummy_input = torch.randn(1, 3, 128, 128, 128, dtype = torch.float32).to(device)
   
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            if device == 'cpu':
                t1_start = perf_counter()
                _,_,_,_ = model(dummy_input)
                t1_stop = perf_counter()
                curr_time = (t1_stop-t1_start)*1000
            else:
                starter.record()
                _,_,_,_ = model(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                
            timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
        
    print(f'=>Model {modelname} inference time: {mean_syn:.2f} std:{std_syn:.2f}. Parameters: {total_params}')      
    return mean_syn, std_syn, total_params

def main():
    
    GPU = True
    CPU = True
    modelnames = ['LSTMDF125','LSTMDF128']#['rPPGNet']
    #modelnames = ['PHYSNET','UBS1','UB64','UB32','UB16','UB8','UB4','UB2']#['PB2','PB1','PHYSNET','P64','PS1','PS2','PS3','UBS1','UB64','UB32','UB16','UB8','UB4','UB2']
    
    if GPU:
        print('GPU:')
    
        df = pd.DataFrame(columns=['model', 'inference', 'std', 'n_params'])
        
        for modelname in modelnames:
            clearGPU(); 
            mean, std, nparams = getInferenceTime(modelname,'auto')
            df = df.append({'model':modelname, 'inference':mean, 'std': std, 'n_params':nparams},ignore_index=True)
            
        print(df)

    if CPU:    
        print('CPU:')
    
        df = pd.DataFrame(columns=['model', 'inference', 'std', 'n_params'])
        
        for modelname in modelnames:
            mean, std, nparams = getInferenceTime(modelname,'cpu')
            df = df.append({'model':modelname, 'inference':mean, 'std': std, 'n_params':nparams},ignore_index=True)
            
        print(df)

if __name__ == '__main__':
    main()