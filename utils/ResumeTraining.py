try:
    if 'google.colab' in str(get_ipython()):
        COLAB = True
    else:
        COLAB = False
except:
    COLAB = False

import os
import sys
import glob
import torch
import numpy as np
from os.path import join, abspath

if not(COLAB):
    from utils.checkpoints import load_checkpoint
    from utils.plotLossHistory import plotLossHistory
else:
    from deep_rppg.utils.checkpoints import load_checkpoint
    from deep_rppg.utils.plotLossHistory import plotLossHistory   

# FUNCTION TO CHECK IF IS POSSIBLE TO RESUME TRAINING
def ResumeTraining(args, model, optimizer, save_path:str, num_epochs:int, Fold:int, VERBOSE:bool):
    '''Function to check if it is possible to resume training in the Experiment object s'''
    save_path = abspath(save_path)
    epoch_zero_resolution = 3
    try:
        # Load model with weights final epoch (it must be the same of num_epochs-1)
        load_checkpoint(model,optimizer,
            join(save_path,str(args.fold),'weights','model_f{}_e{}.pth.tar'.format(Fold,str(num_epochs-1).zfill(epoch_zero_resolution))))
        if VERBOSE>0: print(f'=>[ResumeTraining]: TRAINING OVER. Model trained loaded successfully in last epoch ({num_epochs-1})')

        return True, 0, 0, 0, 0 # If we finish only the flag indicating the end of the training matters.
    except: # If model was not finished try to resume the training or start it
        # Try to resume training from last epoch saved
        if len(os.listdir(join(save_path,str(args.fold),'weights'))) != 0:
            checkpoints_saved = sorted(glob.glob(join(save_path,str(args.fold),'weights',f'model_f{Fold}_e???.pth.tar')))
            load_checkpoint(model,optimizer,checkpoints_saved[-1])
            # resume also losses
            best_validation = torch.load(checkpoints_saved[-1])['best_validation']
            epoch = torch.load(checkpoints_saved[-1])['epoch']+1
            example_ct = torch.load(checkpoints_saved[-1])['example_ct']
            batch_ct = torch.load(checkpoints_saved[-1])['batch_ct']
                              
            if VERBOSE>0: print(f'=>[ResumeTraining]: Epoch {epoch}.')
            return False, epoch, example_ct, batch_ct, best_validation             
        else:   
            if VERBOSE>0: print('=>[ResumeTraining]: Training from scratch.')
            return False, 0, 0, 0, np.inf

def LoadWeightsToModel(args, model, optimizer, save_path:str, filename:str, num_epochs:int, Fold:int, VERBOSE:int):
    '''Function to load specific weights to a model (mostly, the best one)'''
    epoch_zero_resolution = 3
    if filename == 'last':
        try:
            # Load model with weights final epoch (it must be the same of num_epochs-1)
            load_checkpoint(model,optimizer,
                join(save_path,str(args.fold),'weights','model_f{}_e{}.pth.tar'.format(Fold,str(num_epochs-1).zfill(epoch_zero_resolution))))
            if VERBOSE>0: print(f'=>[LoadWeightsToModel]: Model trained loaded successfully in last epoch ({num_epochs-1})')
        except: 
            print('=>[LoadWeightsToModel]: <LAST> There was an error while loading the weights in last epoch')
            
    elif filename == 'best':
        try:
            # Load model with best weights
            bst_weights = sorted(glob.glob(join(save_path,str(args.fold),'weights',f'model_f{Fold}_e???.best.pth.tar')))
            load_checkpoint(model, optimizer, bst_weights[-1])
            if VERBOSE>0: print('=>[LoadWeightsToModel]: <BEST> Model trained loaded successfully: {}'.format(bst_weights[-1]))
        except: 
            print('=>[LoadWeightsToModel]: There was an error while loading the best weights')
            
    else: # If we say the name of some specific weights
        try:
            # Load model with best weights
            load_checkpoint(model, optimizer, join(save_path,str(args.fold),filename))
            if VERBOSE>0: print(f'=>[LoadWeightsToModel]: <filename> Model trained loaded successfully at {join(save_path,str(args.fold),filename)}')
        except: 
            print(f'=>[LoadWeightsToModel]: There was an error while loading {join(save_path,str(args.fold),filename)}')        
    