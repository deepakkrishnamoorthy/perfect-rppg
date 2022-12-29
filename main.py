import os
import argparse
from os.path import join, abspath, exists
import sys
import yaml
from datetime import datetime
import numpy as np
import torch
import wandb
import pandas as pd
from models import MemModule
#from torchsummary import summary

# IMPORT NETWORKS
from networks.PhysNet import PhysNet, P64, PS1, PS2, PS3, PS4, PB1, PB2, UBS1, UB64, UB32, UB16, UB8, UB8_LSTMLAYERS, UB4, UB2, UB64HR
from networks.SOA import rPPGNet
from networks.LSTMDF import LSTMDFMTM125, LSTMDFMTM128, LSTMDFMTO128
# IMPORT DATA MANAGERS
from data.datamanager import DataManager, save_subjects_metadata, getSubjectIndependentTrainValDataFrames, save_windows_metadata
from data import physnet_datasets
from data import lstmdf_datasets
# IMPORT LOSSES
from losses.NP import NP
from losses.NP_NSNR import NP_NSNR
from losses.HR import MAEINHR
from losses.MSE import MSE_lstmdf
# IMPORT METRICS

from metrics.hrmae_hrr_snr import hrmae_hrr_snr
# IMPORT TRAINERS
from trainers import train_eval
# IPMORT TESTERS
from testers import InferenceWithNetworkAndFilter
# IMPORT UTILS
from utils.param_net import DatasetName, NetworkModel, Loss, Optimizer, Metric, ColorChannel
from utils.reproducible import reproducible
from utils.clearGPU import clearGPU
from utils.checkpoints import load_checkpoint
#wandb.init(project="deep_rppg_NIR")

#wandb.init(settings=wandb.Settings(start_method='thread'))
#wandb.init(settings=wandb.Settings(start_method='fork'))
def model_pipeline(args, df, global_VERBOSE):
    
    # GET DATAFRAMES FOR TRAINING AND TESTING SETS WITH SUBJECTS METADATA
    train_df, test_df = getSubjectIndependentTrainValDataFrames(args, df, global_VERBOSE['getTrainValDataFrames'])
    # CREATE MODEL 
    model = build_network(args, args.in_COLAB, args.device, args.network, global_VERBOSE['build_network'])
    # CREATE TRAIN DATASET
    train_ds = build_trainds(args, train_df, global_VERBOSE)
    # BUILD OPTIMIZER
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, global_VERBOSE['build_optimizer'])
    # BUILD CRITERION (LOSS)
    criterion = build_criterion(args,args.loss, args.lambda_loss, global_VERBOSE['build_criterion'] )
    # BUILD METRIC
    metric = build_metric(args.metric, global_VERBOSE['build_metric'])
    # print(model)  
    # IF DEBUGGING REPRODUCIBILITY:
    if args.only_save_train_test_metadata_current_fold:
        print('=>[only_save_train_test_metadata_current_fold] Finish')
        sys.exit()   
        
    # if args.is_SWEEP:
    #     with wandb.init(config=args):
    # else:
        # Create folders output
        # if (not(args.is_SWEEP) and not(exists(join(abspath(args.save_path),str(args.fold))))): os.makedirs(join(abspath(args.save_path),str(args.fold)))
        # resume = 'allow'# bool "allow" ? if: args.is_resume = os.environ["WANDB_RESUME"] = "must", else resume='allow' or resume=True
        # id = 'args.network+args.loss+args.metric+args.batch+'
    
    if args.is_SWEEP:
        # HYPERPARAMETER TUNING
        with wandb.init(config=args):   
                        
            wandb.define_metric('loss_'+args.loss, summary='min')
            wandb.define_metric("hrmae", summary='min')      
            wandb.define_metric("hrr", summary='max')
            wandb.define_metric("snr", summary='max')
          
            if args.network in ['UB64HR']: wandb.define_metric("hrhatmae", summary='min')
    
            # and use them to train the model
            train_eval(args, model, train_ds, test_df, criterion, metric, optimizer, global_VERBOSE)       
        
    else:
        # SINGLE RUN 
        resume_wnb_param = 'must' if args.is_resume else None 
        wandb_mode = 'disabled' if not(bool(args.use_wandb)) else None
        directory = '/content' if args.in_COLAB else None#join(abspath(args.save_path),str(args.fold))
        with wandb.init(mode = wandb_mode,
                        project= args.project_name,
                        name = args.run_name+'_f'+str(args.fold),
                        id = args.run_id,
                        dir = directory,
                        resume = resume_wnb_param,
                        config = args):
            
            wandb.define_metric('loss_'+args.loss, summary='min')
            wandb.define_metric("hrmae", summary='min')      
            wandb.define_metric("hrr", summary='max')
            wandb.define_metric("snr", summary='max')
          
            
            if args.network in ['UB64HR']: wandb.define_metric("hrhatmae", summary='min')
    
            # and use them to train the model
            train_eval(args, model, train_ds, test_df, criterion, metric, optimizer, global_VERBOSE)
    
    return model

def build_network(args,in_COLAB, device, network_name:str, VERBOSE:int):
    '''
    FUNCTION TO CREATE THE MODEL
    in_COLAB (bool): Flag wether in COLAB
    device : cuda or cpu
    network_name (string): name of the network to build
    VERBOSE (bool)
    '''

    if not(in_COLAB==True) and torch.cuda.is_available() : clearGPU(VERBOSE)  
    if network_name == 'PHYSNET':
        network = PhysNet(); 
    elif network_name == 'RPPGNET':
        network = rPPGNet() 
    elif network_name == 'LSTMDFMTM125':
        network = LSTMDFMTM125()  
        network.init_weights()
    elif network_name == 'LSTMDFMTM128':
        network = LSTMDFMTM128()  
        network.init_weights()        
    elif network_name == 'LSTMDFMTO128':
        network = LSTMDFMTO128()  
        network.init_weights()        
    elif network_name == 'P64':
        network = P64()
    elif network_name == 'PS1':
        network = PS1()
    elif network_name == 'PS2':
        network = PS2()
    elif network_name == 'PS3':
        network = PS3()
    elif network_name == 'PS4':
        network = PS4()
    elif network_name == 'PB1':
        network = PB1()
    elif network_name == 'PB2':
        network = PB2()  
    elif network_name == 'UBS1':
        network = UBS1()
    elif network_name == 'UB64':
        network = UB64() 
    elif network_name == 'UB32':
        network = UB32()
    elif network_name == 'UB16':
        network = UB16()
    elif network_name in ['UB8','RTRPPG']:
        network = UB8(args)   
    elif network_name == 'UB8_LSTMLAYERS':
        network = UB8_LSTMLAYERS()          
    elif network_name == 'UB4':
        network = UB4()  
    elif network_name == 'UB2':
        network = UB2()          
    elif network_name == 'UB64HR':
        network = UB64HR()        
    if VERBOSE>0: print(f'=>[make] {network_name} model succesfully created')

    return network.to(device)

def build_trainds(args, train_df, global_VERBOSE):
    '''
    FUNCTION TO CREATE THE TRAIN DATASET
    args (Namespace): Arguments
    traindf (dataFrame): Dataframe with input information
    global_VERBOSE (dict): dictionary with all VERBOSE flags
    '''
    VERBOSE = global_VERBOSE['build_trainds']
        
    # For reproducibility DEBUG: Save first shuffling metadata in training
    if args.save_first_shuffle_metadata: 
        if not(os.path.exists(join(abspath(args.save_path),str(args.fold)))): os.makedirs(join(abspath(args.save_path),str(args.fold)))
        save_windows_metadata(
            is_reproducible = args.is_reproducible,
            in_COLAB = args.in_COLAB,
            load_dataset_path = args.load_dataset_path, 
            save_path = args.save_path,
            config = args,
            train_df = train_df,
            window = args.window,
            step_tr = args.step_tr, 
            db_name = args.database_name,
            seed_dataset = args.seed,
            VERBOSE = global_VERBOSE['save_windows_metadata'])

    if VERBOSE>0: print('=>[make] Creating Train_ds ...')

    if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
        train_ds = lstmdf_datasets.TrainDataset(
            is_reproducible = args.is_reproducible,
            in_COLAB = args.in_COLAB,
            df = train_df,
            window = args.window,
            step = args.step_tr,
            OnlySaveMetaData = False, 
            db_name = args.database_name,
            many_to = args.lstm_many_to,
            seed = args.seed)
        
    else:
    
        train_ds = physnet_datasets.TrainDataset(
            is_reproducible = args.is_reproducible,
            in_COLAB = args.in_COLAB,
            load_dataset_path = args.load_dataset_path,
            df = train_df,
            img_size = args.img_size,
            window = args.window,
            step = args.step_tr,
            HARD_ATTENTION = args.hard_attention,
            OnlySaveMetaData = False, 
            db_name = args.database_name,
            seed = args.seed)
    
    if VERBOSE>0: print('=>[make] Train_ds succesfully created.')    
    
    return train_ds

def build_optimizer(model, optimizer_name:str, learning_rate:float, VERBOSE:int):
    '''
    FUNCTION THE OPTIMIZER
    model(torch.nn.Module): Network to be used
    optmizer_name(str): optimizer's name'
    learning_rate(float): learning rate
    VERBOSE(int)
    '''    

    if optimizer_name == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'ADADELTA':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)

    if VERBOSE>0: print(f'=>[make] {optimizer_name} Optimizer with learning rate {learning_rate} succesfully created')
    
    return optimizer

def build_criterion(args,loss_name:str, Lambda_loss:float, VERBOSE:int):
    '''
    FUNCTION THE OPTIMIZER
    loss_name(str): name of the loss to be created
    Lambda_loss(float): value used in custom loss functions to balance the output
    VERBOSE
    '''  
    assert loss_name in ['NP','NP_NSNR','MSE','MAEINHR'],f'=>[build_criterion][ERROR] loss {loss_name} is not implemented'
    # rPPG losses
    if loss_name == 'NP':
        criterion = NP()
        if VERBOSE>0: print(f'=>[build_criterion] {loss_name} criterion succesfully created') 
    elif loss_name == 'NP_NSNR':
        criterion = NP_NSNR(Lambda_loss)
        if VERBOSE>0: print(f'=>[build_criterion] {loss_name} with lambda {args.lambda_loss} criterion succesfully created') 
    elif loss_name == 'MSE':
        criterion = MSE_lstmdf()
        if VERBOSE>0: print(f'=>[build_criterion] {loss_name} criterion succesfully created') 
    # HR losses
    elif loss_name == 'MAEINHR':
        criterion = MAEINHR()
        if VERBOSE>0: print(f'=>[build_criterion] {loss_name} criterion succesfully created') 
    

    return criterion    

def build_metric(metric_name:str, VERBOSE):
    """
    CREATE METRIC
    metric_name(str): name of the metric to be used: todo: use more metrics?
    VERBOSE
    """ 
    metric = hrmae_hrr_snr()
    if VERBOSE>0: print(f'=>[make] Using {metric_name} as validation criterion for save best checkpoints') 
    
    return metric    
    
def main_exp(dict_args=None):

    print('================================================================')
    print('                     DEPP_RPPG                                  ')
    print('================================================================')     

    """""""""
    DEFINE VERBOSE VALUES MANUALLY
    """""""""
    
    global_VERBOSE={'DataManager':1,  
             'save_subjects_metadata':1,
             'getTrainValDataFrames':1,
             'save_windows_metadata':1,
             'train_eval':1,
             'ResumeTraining':1,
             'clearGPU':1,
             'evaluate':1,
             'LoadWeightsToModel':1,
             'build_network':1,
             'build_trainds':1,
             'build_optimizer':1,
             'build_criterion':1,
             'build_metric':1}
    
    if dict_args == None:
    
        """""""""
        START ARGPARSE
        """""""""
        parser = argparse.ArgumentParser()
        # EXPERIMENTAL SETUP  
        parser.add_argument('--use_wandb', type=int, choices=[0,1], default=1,required=False)        
        parser.add_argument('--in_COLAB', type=int, choices=[0,1], default=1,required=True)
        parser.add_argument('--is_SWEEP', type=int, choices=[0,1], default=0, required=False) # Only for sweep hyperparameter tuning
        parser.add_argument('-ne', '--project_name', type=str, required=False)    
        parser.add_argument('-rn', '--run_name', type=str, required=False)
        parser.add_argument('-rid', '--run_id', type=str, required = False)
        parser.add_argument('--is_resume', type=int, choices=[0,1], default=0)    
        parser.add_argument('--is_reproducible', type=int, choices=[0,1], default=1)
        parser.add_argument('-s_r', '--seed', type=int, default=10, required=False)          
        parser.add_argument('-lp', '--load_dataset_path', type=str, required=True)
        parser.add_argument('-sp', '--save_path', type=str, required=True)
        parser.add_argument('-d', '--database_name', type=str,
                            choices=[e.name for e in DatasetName], required=True)
        parser.add_argument('-ds_p', '--dataset_percentage', type=float, default=1)        
        parser.add_argument('-f', '--fold', type=int,
                            choices=[0,1,2,3,4,-1,-2], required=True)#-1=train_test_split(80/20),-2=Train in all data          
        parser.add_argument('-ch', '--channels', type=str,
                            choices=[e.name for e in ColorChannel], default='YUV') 
        parser.add_argument('-isz', '--img_size', type=int, choices=[2,4,8,16,32,64,128], default=64)
        parser.add_argument('-sme', '--save_model_each_n_epochs', type=int, default=1)
        parser.add_argument('--is_5050_validation', type=int, choices=[0,1], default=0)
        parser.add_argument('--use_data_augmentation', type=int, choices=[0,1], default=0)
        parser.add_argument('--hard_attention', type=int, choices=[0,1], default=0)
        
        # LSTM SETUP
        parser.add_argument('--lstm_state',type=str,choices=['stateful','stateless'],default='stateful')
    
        # HYPERPARAMETERS
        parser.add_argument('-opt', '--optimizer', type=str,
                            choices=[e.name for e in Optimizer], required=True)
        parser.add_argument('-l', '--loss', type=str,
                            choices=[e.name for e in Loss], required=True)      
        parser.add_argument('-m', '--metric', type=str,
                            choices=[e.name for e in Metric], required=True, default='hrmae')     
        parser.add_argument('-n', '--network', type=str,
                            choices=[e.name for e in NetworkModel], required=True)
        parser.add_argument('-nep', '--epochs', type=int, default=15, required=False)
        parser.add_argument('-bs', '--batch_size', type=int, default=8, required=True) 
        parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, required=True) 
        parser.add_argument('-la', '--lambda_loss', type=float, default=1, required=False)
    
        #           FOR PHYSNET AND PHYSNET-BASED
        parser.add_argument('-w', '--window', type=int,
                        choices=[i for i in range(1,129)], default=128) 
        parser.add_argument('-st', '--step_tr', type=int,
                            choices=[i for i in range(1,129)], default=128) 
        parser.add_argument('-se', '--step_eval', type=int,
                            choices=[1,125,128], default=128)   
    
        # TOOLS FOR ONLY PREDICTION and CROSS-DATASET
        parser.add_argument('--is_prediction', type=int, choices=[0,1], default=0)
        parser.add_argument('--overwrite_prediction', type=int, choices=[0,1], default=0)
        parser.add_argument('--use_these_weights', type=str, required=False, choices=['best','last'],default='best')
        parser.add_argument('--predict_without_training', type=int, choices=[0,1], default=0)
        parser.add_argument('--load_these_weights', type=str, default='None', required=False) # Cross dataset
     
        # TOOLS TO SAVE METADATA TO DEBUG REPRODUCIBILITY
        parser.add_argument('-ssm', '--save_subjects_metadata', type=int, default=0, required=False)#1=full data,2=full data + train + val    
        parser.add_argument('--save_first_shuffle_metadata', type=int, choices=[0,1], default=0)#To compare reproducibility in training shuffling   
        parser.add_argument('--only_save_train_test_metadata_current_fold', type=int, choices=[0,1], default=0)  
        #parser.add_argument('--mem_dim', type=int, choices=[2000,3000], default=2000)#mewly-added
    
        # TOOLS TO DEBUG NETWORK
    
    
        # TOOLS TO MANAGE DATA ONLY IN COLAB    
        parser.add_argument('--only_data_manager', type=int, choices=[0,1], default=0)# Only to import data from DRive to COLAB VM
        
        # JOIN ALL ARGUMENTS
        args = parser.parse_args()
 
        # UPDATE TYPE FROM INT TO BOOL IF NECESSARY (CUZ IT IS NOT POSSIBLE DIRECTLY)
        args.use_wandb = bool(args.use_wandb)
        args.is_SWEEP = bool(args.is_SWEEP)
        args.is_resume = bool(args.is_resume)
        args.is_reproducible = bool(args.is_reproducible)   
        args.is_5050_validation = bool(args.is_5050_validation) 
        args.use_data_augmentation = bool(args.use_data_augmentation) 
        args.hard_attention = bool(args.hard_attention)   
        args.is_prediction = bool(args.is_prediction)
        args.overwrite_prediction = bool(args.overwrite_prediction)
        args.predict_without_training = bool(args.predict_without_training)
        args.save_first_shuffle_metadata = bool(args.save_first_shuffle_metadata)
        args.only_save_train_test_metadata_current_fold = bool(args.only_save_train_test_metadata_current_fold)
        args.only_data_manager = bool(args.only_data_manager)
    
    else:
        """""""""
        CREATE ARGS FROM DICTIONARY
        """""""""
        args = argparse.Namespace()
        args.is_SWEEP = bool(dict_args['is_SWEEP'])
        args.project_name = dict_args['project_name']
        args.run_name = dict_args['run_name']
        args.run_id = dict_args['run_id']
        args.is_resume = bool(dict_args['is_resume'])
        args.load_dataset_path = dict_args['load_dataset_path']
        args.save_path = dict_args['save_path']
        args.is_reproducible = bool(dict_args['is_reproducible'])
        args.seed = dict_args['seed']
        args.database_name = dict_args['database_name']
        args.channels = dict_args['channels']
        args.dataset_percentage = dict_args['dataset_percentage']
        args.save_model_each_n_epochs = dict_args['save_model_each_n_epochs']
        args.is_5050_validation = bool(dict_args['is_5050_validation'])
        args.use_data_augmentation = bool(dict_args['use_data_augmentation'])
        args.img_size = dict_args['img_size']
        args.hard_attention = bool(dict_args['hard_attention'])
        args.fold = dict_args['fold']
        args.optimizer = dict_args['optimizer']
        args.loss = dict_args['loss']
        args.metric = dict_args['metric']
        args.network = dict_args['network']
        args.epochs = dict_args['epochs']
        args.batch_size = dict_args['batch_size']
        args.learning_rate = dict_args['learning_rate']
        args.lambda_loss = dict_args['lambda_loss']
        args.window = dict_args['window']
        args.step_tr = dict_args['step_tr']
        args.step_eval = dict_args['step_eval']
        args.is_prediction = bool(dict_args['is_prediction'])
        args.overwrite_prediction = bool(dict_args['overwrite_prediction'])
        args.use_these_weights = str(dict_args['use_these_weights'])
        args.predict_without_training = bool(dict_args['predict_without_training'])
        args.save_subjects_metadata = int(dict_args['save_subjects_metadata'])
        args.save_first_shuffle_metadata = bool(dict_args['save_first_shuffle_metadata'])
        args.only_save_train_test_metadata_current_fold = bool(dict_args['only_save_train_test_metadata_current_fold'])
        args.only_data_manager = bool(dict_args['only_data_manager'])
        args.use_wandb = bool(dict_args['use_wandb'])
    
    # Check if working with rPPG signals instead of images as input
    if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
        args.is_from_rPPG = True # Our input will be rPPG instead of images
        args.lstm_many_to = 'MTM' if args.network.find('MTM')!=-1 else 'MTO'
        if args.lstm_many_to in ['MTO'] and args.step_eval != 1:
            args.step_eval=1 # When Using MTO there is not other choice than using step_eval=1
            print(f'=>[WARNING!] when using {args.network}, step_eval must be 1 because MTO. step_eval set to 1')
    else:
        args.is_from_rPPG = False
    #args.in_COLAB = os.path.isdir(r'/usr/local/lib/python3.7/dist-packages/google/cloud')
    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # check W&B ID
    if (not(args.is_SWEEP) and args.run_id == None and args.use_wandb):
        assert not(args.is_resume), print('=>[ERROR] if --is_resume, you must give --run_id')
        args.run_id = wandb.util.generate_id()

    """""""""
    MANAGING POSSIBLE BUGS
    """""""""  
    # check inputsize for network
    if args.network in ['RPPGNET']:
        assert(args.img_size==128), f'=>[INPUT ERROR]:{args.network} must use 128x128 images'
    elif args.network in ['P64','UB64','UB64HR']:
        assert(args.img_size==64), f'=>[INPUT ERROR]:{args.network} must use 64x64 images'
    elif args.network in ['UB32']:        
        assert(args.img_size==32), f'=>[INPUT ERROR]:{args.network} must use 32x32 images'
    elif args.network in ['UB16']:        
        assert(args.img_size==16), f'=>[INPUT ERROR]:{args.network} must use 16x16 images'   
    elif args.network in ['UB8','RTRPPG','RTRPPG_LSTMDFMTM128','RTRPPG_LSTMDFMTO128']:        
        assert(args.img_size==8), f'=>[INPUT ERROR]:{args.network} must use 8x8 images' 
    elif args.network in ['UB4']:        
        assert(args.img_size==4), f'=>[INPUT ERROR]:{args.network} must use 4x4 images' 
    elif args.network in ['UB2']:        
        assert(args.img_size==2), f'=>[INPUT ERROR]:{args.network} must use 2x2 images'         

    # Check irregularities if SWEEP
    if args.is_SWEEP:
        assert(not(args.is_prediction)), '[INPUT ERROR]: is_prediction not valid when is_SWEEP is True'
        assert(not(args.overwrite_prediction)), '[INPUT ERROR]: overwrite_prediction not valid when is_SWEEP is True'
        assert(not(args.predict_without_training)), '[INPUT ERROR]: predict_without_training not valid when is_SWEEP is True'
        assert(not(args.is_resume)), '[INPUT ERROR]: is_resume not valid when is_SWEEP is True'

    # Check irregularities if rPPGNet network is used
    if args.network in ['RPPGNET']:
        assert args.channels in ['RGB'], f'=>[COLOR CHANNEL ERROR]:{args.network} must use RGB channels'
        assert(args.window==64), f'=>[WINDOW SIZE ERROR]:{args.network} must use 64-size windows'
        assert(args.step_tr==64), f'=>[STEP TRAINING ERROR]:{args.network} must use 64-step during training'      
    # Check irregularities if UB64HR network is used
    elif args.network in ['UB64HR']:
        assert(args.network in ['UB64HR'] and args.loss == 'MAEINHR'), f'=>[LOSS ERROR]:{args.network} must use loss MAEINHR'
        assert(args.network in ['UB64HR'] and args.metric == 'hrhatmae'), f'=>[METRIC ERROR]:{args.metric} should use loss hrhatmae'
    # Check irregularities if LSTMDF128 network is used
    elif args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
        assert args.is_from_rPPG == 1, f'=>[ERROR]:{args.network} works only with --is_from_rPPG set to 1' 
        if args.network in ['LSTMDFMTM125']:
            assert args.window == 125, f'=>[ERROR]:{args.network} works only with --window set to 125'
        elif args.network in ['LSTMDFMTM128','LSTMDFMTO128']:
            assert args.window == 128, f'=>[ERROR]:{args.network} works only with --window set to 128'        

    """""""""
    SHOW PARAMETERES CHOOSEN FOR THIS EXPERIMENT
    """""""""  
    if not(args.is_SWEEP):
        for arg in vars(args):
            print(f'{arg} : {getattr(args, arg)}')
        print('================================================================')
        print('================================================================')         
    
    """""""""
    SET EXPERIMENT REPRODUCIBILITY
    """""""""          
    if args.is_reproducible:        
        reproducible(args.seed)
 
    """""""""
    DATA MANAGMENT
    """""""""   
           
    # CREATE DATAFRAME WITH THE SUBJECTS THAT WILL BE USED IN THE EXPERIMENT
    df = DataManager(args,args.load_dataset_path, args.database_name, args.is_reproducible, args.seed, args.dataset_percentage, global_VERBOSE['DataManager'],is_COLAB=args.in_COLAB)
    if args.only_data_manager:
        print('=>[only_data_manager] Finish')
        sys.exit()

    # SAVE SUBJECTS TO BE USED    
    if args.save_subjects_metadata>=2:
        save_subjects_metadata(args,df,args.save_path,args.load_dataset_path,VERBOSE=global_VERBOSE['save_subjects_metadata'],is_COLAB=args.in_COLAB)

    # Set Wandb folder
    #os.environ["WANDB_DIR"]=os.path.join(abspath(args.save_path),args.fold)
    """""""""
    BUILD, TRAIN AND ANALYZE THE MODEL WITH THE PIPELINE
    """""""""     
    if args.network in ['RTRPPG_LSTMDFMTM128','RTRPPG_LSTMDFMTO128']:
        InferenceWithNetworkAndFilter(args,df=df)
        return
    # Build, train and analyze the model with the pipeline

    model = model_pipeline(args, df=df, global_VERBOSE=global_VERBOSE)
    
    # l2_lambda = 0.01
    # l2_reg = torch.tensor(0.)
    
    # for param in model.parameters():
    #     l2_reg += torch.norm(param)
    # loss += l2_lambda * l2_reg

    # for param in model.parameters():
    #     l2_reg += torch.norm(param)

    #     loss += l2_lambda * l2_reg
    
    #  

    
    # for param in model.ConvBlock1.parameters():
    #     param.requires_grad = False
        
    # for param in model.ConvBlock2.parameters():
    #     param.requires_grad = False
    
    
    #for para in model.parameters():
     #   print(para)
    #for param in model.parameters():
        #param.requires_grad = False
    #for param in model.features[2].parameters():
        #param.requires_grad = False
    
    # for param in model.ConvBlock1.parameters():
    #     param.requires_grad = False
        
    # for param in model.ConvBlock2.parameters():
    #     param.requires_grad = False
        
    # for param in model.ConvBlock2.parameters():
    #     param.requires_grad = False
        
    # for param in model.ConvBlock3.parameters():
    #      param.requires_grad = False
  
if __name__ == "__main__":
    main_exp()