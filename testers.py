try:
    if 'google.colab' in str(get_ipython()):
        COLAB = True
    else:
        COLAB = False
except:
    COLAB = False

import os
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from os.path import join, abspath
import scipy.io as sio
import PIL
from scipy.signal import savgol_filter
from sklearn.metrics import mean_absolute_error

from scipy import interpolate
import glob
import matplotlib.pyplot as plt
#import plotly.express as px
#from plotly.offline import plot as plotoff
import pandas as pd

if not(COLAB):
    from utils.pearsonr import r
    from utils.getHRandSNR import getHRandSNR    
    from utils.normalize import normalize
    from utils.clearGPU import clearGPU
    from utils.filters import butter_bandpass_filter
    from utils.phase_align import phase_align
    from data import physnet_datasets
    from data import lstmdf_datasets
    # IMPORT NETWORKS
    from networks.PhysNet import UB8
    from networks.LSTMDF import LSTMDFMTM128, LSTMDFMTO128
    # IMPORT DATA MANAGERS
    from data.datamanager import getSubjectIndependentTrainValDataFrames
    from data import physnet_datasets
    from data import lstmdf_datasets
else:
    from deep_rppg.utils.pearsonr import r
    from deep_rppg.utils.getHRandSNR import getHRandSNR    
    from deep_rppg.utils.normalize import normalize
    from deep_rppg.utils.clearGPU import clearGPU 
    from deep_rppg.utils.filters import butter_bandpass_filter
    from deep_rppg.utils.phase_align import phase_align
    from deep_rppg.data import physnet_datasets 
    from deep_rppg.data import lstmdf_datasets
    # IMPORT NETWORKS
    from deep_rppg.networks.PhysNet import UB8
    from deep_rppg.networks.LSTMDF import LSTMDFMTM128, LSTMDFMTO128
    # IMPORT DATA MANAGERS
    from deep_rppg.data.datamanager import getSubjectIndependentTrainValDataFrames
    from deep_rppg.data import physnet_datasets
    from deep_rppg.data import lstmdf_datasets
    
def evaluate(args, model, criterion, test_df, window, step_eval, epoch, example_ct, is_Predict, global_VERBOSE):
    
    VERBOSE = global_VERBOSE['evaluate']
    if not(is_Predict):
        pass
        # if VERBOSE>0: print('=>[evaluate] Test evaluation')
    else:
        assert step_eval==1, print(f'=>[evaluate] ERROR: Using step_eval={step_eval} during Prediction is not logic. Please use step_eval = 1.')
        if VERBOSE>0: print(f'=>[evaluate] Saving test Prediction in {args.save_path}')
 
    HR_y_hat, HR_y, SNR_y_hat = np.array([]),np.array([]),np.array([])
    if args.network in ['UB64HR']: HRmae_hat = np.array([])#Save HRMAE given by the network HR, not the one measured by FFT in rPPG
    # ITERATE SUBJECT BY SUBJECT

    for sbj_idx, subject_df in test_df.iterrows(): # For al individual subjects
        if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
            subject_ds = lstmdf_datasets.SubjectIndependentTestDataset(
                in_COLAB = args.in_COLAB,
                subject_df = subject_df,
                window = window,
                step = step_eval,
                db_name = args.database_name,
                many_to = args.lstm_many_to,
              )
            
        else:
            subject_ds = physnet_datasets.SubjectIndependentTestDataset(
                in_COLAB = args.in_COLAB,
                load_dataset_path =args.load_dataset_path,
                subject_df = subject_df,
                img_size = args.img_size,
                window = window,
                step = step_eval,
                HARD_ATTENTION = args.hard_attention,
                db_name = args.database_name
              )

        # subject_ds.plot_first_middle_last_sample(0)
        test_loader = DataLoader(subject_ds, args.batch_size, drop_last=False, shuffle=False)
        name = subject_ds.name # subject name
        x_orig = subject_ds.x_file
        GT = subject_ds.y_file # GT
        time = subject_ds.t_file # time
        rPPG = []

        # If overwritePred=True start from first subject, else, resume in last prediction
        if is_Predict:
            if args.overwrite_prediction==False:
                try:#Check if current subject already exist
                    crt_sbjt = sio.loadmat(join(args.save_path,str(args.fold),name.split('.')[0]+'.mat'))
                    continue
                except:
                    pass
            
        with torch.no_grad():
            if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128'] and args.lstm_state in ['stateful']:
                model.reset_hidden()  
            #model.eval()
            for idx, sample in enumerate(test_loader):
                # Forward
                if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
                    if args.lstm_state in ['stateless']:
                        model.reset_hidden() 
                    out,_,_,_ = model(sample['x'])
                    out = out.squeeze(-1)
                else: #PHYSNET-like models
                    out, HR_h, _, _ = model(sample['x'])
                    out = out-torch.mean(out,keepdim=True,dim=1)/torch.std(out,keepdim=True,dim=1)   
                    
                rPPG.append(out.to('cpu').detach().numpy())
                if args.network in ['UB64HR']:
                    loss = criterion([sample['x'], sample['y'], sample['t'], HR_h, {'is_eval':True}])
                    HRmae_hat = np.append(HRmae_hat,loss.to('cpu').detach().numpy())
                
        rPPG = np.vstack(rPPG)
        # Do OVERLAPP-ADD only if step_eval = 1        
        if step_eval==1:  
            if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
                if args.lstm_many_to == 'MTO':
                    y_hat = np.concatenate((normalize(np.expand_dims(x_orig[0:args.window],-1)),normalize(rPPG)),axis=0)
                    y_hat = y_hat.flatten()
                elif args.lstm_many_to == 'MTM':
                    ## OVERLAP-ADD PROCESS                
                    y_hat = np.zeros(window+len(rPPG)-1)
                    for i in range(len(rPPG)):#For all sub-samples of X_val_i signal
                        y_hat[i:i+window] = y_hat[i:i+window]+rPPG[i]
                    y_hat = np.squeeze(normalize(y_hat))
            else:
                ## OVERLAP-ADD PROCESS                
                y_hat = np.zeros(window+len(rPPG)-1)
                for i in range(len(rPPG)):#For all sub-samples of X_val_i signal
                    y_hat[i:i+window] = y_hat[i:i+window]+rPPG[i]
                y_hat = np.squeeze(normalize(y_hat))
            
        else: # ONLY FOR EVALUATION when step_eval = 128
            y_hat = rPPG.flatten()
            # If missing frames just stack the ground truth ones
            if len(y_hat)<len(GT):                
                y_hat = np.concatenate((np.squeeze(normalize(y_hat)),GT[len(y_hat):]))
        
        if is_Predict: 
            # SAVE PREDICTIONS WITH INPUT AND GROUND TRUTH
            if args.network.find('LSTM') != -1: # If we are using any LSTM we want to compare with other filters
                # BANDPASS FILTER
                # https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html                   
                fs = int(1/np.mean(np.diff(time))) # Sampling frequency from time vector
                BP = butter_bandpass_filter(x_orig, lowcut=0.7, highcut=3.5, fs=25, order=8)
                # BP alignment with RPPG         
                ROI_ini = 0 # begin in 0 seconds
                ROI_end = ROI_ini+(5*fs) # finish 5 seconds latter        
                s2 = phase_align(x_orig, BP, [ROI_ini,ROI_end])# align by phase
                BP = np.roll(BP, int(s2))#      
                # SAVGOLAY FILTER
                SG = savgol_filter(x_orig, window_length=9, polyorder=2)#wavelet_filter(X_val[i], 0.4,wavelet='db4')
                if len(SG)!=len(x_orig):#If the ouput doesnt have the same length cut or add a value
                    if len(SG)>len(x_orig):
                        SG = SG[0:-1]
                    else:
                        SG = np.append(SG,SG[-1])
                #plt.figure(),plt.title('Input'),plt.plot(time,GT,'r'),plt.plot(time,x_orig)
                #plt.figure(),plt.title('Input+LSTM'),plt.plot(time,GT,'r'),plt.plot(time,y_hat)
                #plt.figure(),plt.title('Input+BP'),plt.plot(time,GT,'r'),plt.plot(time,BP)
                #plt.figure(),plt.title('Input+SG'),plt.plot(time,GT,'r'),plt.plot(time,SG)
                if VERBOSE>0: print('=>[PREDICTION]: Saving '+ join(args.save_path,str(args.fold),name.split('.')[0]+'.mat'))
                sio.savemat(join(args.save_path,str(args.fold), name.split('.')[0]+'.mat'),
                            {'rPPG': x_orig,
                             'BP': BP,
                             'SG': SG,
                             'GT': GT,
                             'Pred': y_hat,
                             'timeTrace': time})                        
       
            else: # In other cases just save te prediction
            
                if VERBOSE>0: print('=>[PREDICTION]: Saving '+ join(args.save_path,str(args.fold),name.split('.')[0]+'.mat'))
                sio.savemat(join(args.save_path,str(args.fold), name.split('.')[0]+'.mat'),
                            {'Pred': y_hat,
                              'GT': GT,
                              'timeTrace': time})
            
        else: # If Evaluation only
        
            if sbj_idx==0: # Log first prediction
                try:
                  if args.in_COLAB:
                      fig, ax = plt.subplots()
                      ax.plot(time,GT), ax.plot(time,y_hat)
                      ax.legend(['GT','Pred'])
                      wandb.log({f"val_pred{epoch}": fig})
                  else:
                      fig, ax = plt.subplots()
                      ax.plot(time,GT), ax.plot(time,y_hat)
                      ax.legend(['GT','Pred'])
                      wandb.log({f"val_pred{epoch}": fig})
                      plt.close('all')
                except:
                    print('=>[evaluate] There was an error when trying to plot the first prediction in epoch {epoch}')

            if args.database_name in ['VIPL','ECGF','BIGECGF','UBFC','BIGUBFC','COHFACE'] and not(args.is_from_rPPG):# Resample to 25 Hz for a fair comparison            
                if len(time)-len(y_hat) == 1 : # If there is 1 value length difference...
                    if len(time)>len(y_hat): time = time[:-1]
                    if len(y_hat)>len(time): y_hat = y_hat[:-1]
                    
                timeTrace25hz = np.arange(0,time[-1],(1./25.))
                f = interpolate.interp1d(time, y_hat)
                y_hat = f(timeTrace25hz)  
                f = interpolate.interp1d(time, GT)
                GT = f(timeTrace25hz) 
            
            # Get HR in y_hat and y along SNR_y_hat to get Metrics in current subject
            HR_y_hat_i, HR_y_i, SNR_y_hat_i = getHRandSNR(y_hat, GT, Fs=25, winLengthSec=15, stepSec=0.5, lowF=0.7, upF=3.5, VERBOSE=0)
            if len(HR_y_hat_i)>0: HR_y_hat = np.concatenate((HR_y_hat,HR_y_hat_i))
            if len(HR_y_i)>0: HR_y = np.concatenate((HR_y,HR_y_i))
            if len(SNR_y_hat_i)>0: SNR_y_hat = np.concatenate((SNR_y_hat,SNR_y_hat_i))

    # At this point the prediction was done in all subjects     
    if not(is_Predict): # If is evaluation, log the metris average and return hrmae      
        if not(HR_y_hat.size==0 or HR_y.size==0 or SNR_y_hat.size==0):
            hrmae = mean_absolute_error(np.array(HR_y_hat), np.array(HR_y))
            hrr = r(HR_y_hat, HR_y)
            snr = np.average(SNR_y_hat)
        else:
            print("=>[evaluate] WARNING! Empty metrics when using getHRandSNR")
            hrmae = 1000
            hrr = -1
            snr = -1000
        
        wandb.log({"epoch": epoch, 'val/hrmae': hrmae})
        wandb.log({"epoch": epoch, 'val/hrr': hrr})
        wandb.log({"epoch": epoch, 'val/snr': snr})
        
        if args.network in ['UB64HR']:
            hrhatmae = HRmae_hat.mean()
            wandb.log({"epoch": epoch, 'val/hrhatmae': hrhatmae})
            return hrhatmae
        
        return hrmae

# FUNCTION TO GET THE TIMETRACE OF A SPECIFIC SUBJECT
def GetTimeTraceCrtSubject(y_hat, PathL_timeFile, in_COLAB, db_name:str):
    '''
    Function to load or generate timeTrace vector for y_hat. We create it with the
    sampling rate and in VIPL we have to take the one given by the authors
    Args:
        y_hat (vector): rPPG signal predicted (only the size is important)
        PathL_timeFile (string): Path where is the "pXvXsX_timestamp.txt"
        db_name(str): database name
    '''
    if in_COLAB==True:
        PathL_timeFile = r'/content/data'
        
    if db_name in 'VIPL':
        file_name = glob.glob(join(PathL_timeFile,'*_timestamp.txt'))
        timeTrace = np.loadtxt(file_name[0])
    elif db_name in 'MMSE': # Always 25 Hz
        timeTrace = np.arange(0,(1./25.)*len(y_hat),(1./25.)) 
    elif db_name in ['ECGF','VIPL_ECGF']: #Always 30 Hz
        timeTrace = np.arange(0,(1./30.)*len(y_hat),(1./30.)) 
        
    return timeTrace 

def InferenceWithNetworkAndFilter(args,df):
    '''
    2022/04/21
    This function was created to do the inference of subjects when RTRPPG was trained and LSTMDFMTM128
    I did not want to change te global code so I decided to use a new function. 
    Here args.save_path must have 5 folders: 0,1,2,3,4 and inside of each of them a folder called 'weights'
    with the weights of the RTRPPG and LSTMDF per fold. ex: rtrppg_f0_e014.best.pth.tar and lstmdfmtm128_f0_e035.best.pth.tar
    
    '''
    for fold in [0]:#fold in [0,1,2,3,4]:
        args.fold = fold
        VERBOSE = 1
        window = args.window
        # GET DATAFRAMES FOR TRAINING AND TESTING SETS WITH SUBJECTS METADATA
        train_df, test_df = getSubjectIndependentTrainValDataFrames(args, df, True)
        # CREATE MODEL 
        if not(args.in_COLAB==True) and torch.cuda.is_available(): clearGPU(True)  
        model =  UB8(); model.to(args.device)
        if args.network.find('LSTMDFMTM')!=-1:
            modelfilter = LSTMDFMTM128(); modelfilter.to(args.device)
            modelfilter.init_weights() 
        elif args.network.find('LSTMDFMTO')!=-1:
            modelfilter = LSTMDFMTO128(); modelfilter.to(args.device)
            modelfilter.init_weights() 
        else:
            print('=>[InferenceWithNetworkAndFilter] Filter network does not exist')
            return
        # Load weights to models
        checkpoints_list = os.listdir(join(args.save_path,str(args.fold),'weights'))
        if len(checkpoints_list)==2:
            if checkpoints_list[0].find('rtrppg')!=-1:
                checkpoint = torch.load(join(args.save_path,str(args.fold),'weights',checkpoints_list[0]))
                model.load_state_dict(checkpoint['model_state_dict'])            
                checkpoint = torch.load(join(args.save_path,str(args.fold),'weights',checkpoints_list[1]))
                modelfilter.load_state_dict(checkpoint['model_state_dict'])
                print('=>[InferenceWithNetworkAndFilter]: Checkpoints loaded succesfully') 
            else:
                checkpoint = torch.load(join(args.save_path,str(args.fold),'weights',checkpoints_list[1]))
                model.load_state_dict(checkpoint['model_state_dict'])            
                checkpoint = torch.load(join(args.save_path,str(args.fold),'weights',checkpoints_list[0]))
                modelfilter.load_state_dict(checkpoint['model_state_dict'])
                print('=>[InferenceWithNetworkAndFilter]: Checkpoints loaded succesfully')           
        else:
            print('=>[InferenceWithNetworkAndFilter]: There must be 2 checkpoints files')
            return
        
        # PREDICTION - INFERENCE
        for sbj_idx, subject_df in test_df.iterrows(): # For al individual subjects
            subject_ds = physnet_datasets.SubjectIndependentTestDataset(
                in_COLAB = args.in_COLAB,
                load_dataset_path =args.load_dataset_path,
                subject_df = subject_df,
                img_size = args.img_size,
                window = args.window,
                step = 1,
                HARD_ATTENTION = args.hard_attention,
                db_name = args.database_name
              )
            
            # subject_ds.plot_first_middle_last_sample(0)
            test_loader = DataLoader(subject_ds, args.batch_size, drop_last=False, shuffle=False)
            name = subject_ds.name # subject name
            x_orig = subject_ds.x_file
            GT = subject_ds.y_file # GT
            time = subject_ds.t_file # time
            rPPG = []
            rPPG_LSTM = []
    
            # If overwritePred=True start from first subject, else, resume in last prediction
            if args.overwrite_prediction==False:
                try:#Check if current subject already exist
                    crt_sbjt = sio.loadmat(join(args.save_path,str(args.fold),name.split('.')[0]+'.mat'))
                    continue
                except:
                    pass
            
            with torch.no_grad():
                modelfilter.reset_hidden()# Stateful 
                #model.eval()
                #modelfilter.eval()
                for idx, sample in enumerate(test_loader):
                    # Forward
                    out, _, _, _ = model(sample['x'])
                    out = out-torch.mean(out,keepdim=True,dim=1)/torch.std(out,keepdim=True,dim=1)  
                    rPPG.append(out.to('cpu').detach().numpy())
                    out,_,_,_ = modelfilter(out.unsqueeze(-1))
                    out = out.squeeze(-1)
                    rPPG_LSTM.append(out.to('cpu').detach().numpy())
                    
                rPPG = np.vstack(rPPG)
                rPPG_LSTM = np.vstack(rPPG_LSTM)
    
                ## OVERLAP-ADD PROCESS                
                y_hat = np.zeros(window+len(rPPG)-1)
                for i in range(len(rPPG)):#For all sub-samples of X_val_i signal
                    y_hat[i:i+window] = y_hat[i:i+window]+rPPG[i]
                y_hat = np.squeeze(normalize(y_hat))
    
                ## OVERLAP-ADD PROCESS                
                y_hat_LSTM = np.zeros(window+len(rPPG_LSTM)-1)
                for i in range(len(rPPG_LSTM)):#For all sub-samples of X_val_i signal
                    y_hat_LSTM[i:i+window] = y_hat_LSTM[i:i+window]+rPPG_LSTM[i]
                y_hat_LSTM = np.squeeze(normalize(y_hat_LSTM))
    
                # BANDPASS FILTER
                # https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html                   
                fs = int(1/np.mean(np.diff(time))) # Sampling frequency from time vector
                BP = butter_bandpass_filter(y_hat, lowcut=0.7, highcut=3.5, fs=25, order=8)
                # BP alignment with RPPG
                ROI_ini = 0 # begin in 0 seconds
                ROI_end = ROI_ini+(5*fs) # finish 5 seconds latter        
                s2 = phase_align(y_hat, BP, [ROI_ini,ROI_end])# align by phase
                BP = np.roll(BP, int(s2))#      
                # SAVGOLAY FILTER
                SG = savgol_filter(y_hat, window_length=9, polyorder=2)#wavelet_filter(X_val[i], 0.4,wavelet='db4')
                if len(SG)!=len(y_hat):#If the ouput doesnt have the same length cut or add a value
                    if len(SG)>len(y_hat):
                        SG = SG[0:-1]
                    else:
                        SG = np.append(SG,SG[-1])
                #plt.figure(),plt.title('Input'),plt.plot(time,GT,'r'),plt.plot(time,y_hat)
                #plt.figure(),plt.title('Input+LSTM'),plt.plot(time,GT,'r'),plt.plot(time,y_hat_LSTM)
                #plt.figure(),plt.title('Input+BP'),plt.plot(time,GT,'r'),plt.plot(time,BP)
                #plt.figure(),plt.title('Input+SG'),plt.plot(time,GT,'r'),plt.plot(time,SG)
                if VERBOSE>0: print('=>[PREDICTION]: Saving '+ join(args.save_path,str(args.fold),name.split('.')[0]+'.mat'))
                sio.savemat(join(args.save_path,str(args.fold), name.split('.')[0]+'.mat'),
                            {'rPPG': y_hat,
                             'BP': BP,
                             'SG': SG,
                             'GT': GT,
                             'Pred': y_hat_LSTM,
                             'timeTrace': time})                        
