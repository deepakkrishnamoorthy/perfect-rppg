try:
    if 'google.colab' in str(get_ipython()):
        COLAB = True
    else:
        COLAB = False
except:
    COLAB = False
    
import torch
import os
from os.path import join, abspath, exists
from torch.utils.data import DataLoader
import numpy as np
import wandb
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
#import plotly.express as px
#from plotly.offline import plot as plotoff
import pandas as pd
import PIL

if not(COLAB):
    from utils.plotLossHistory import plotLossHistory
    from utils.checkpoints import save_checkpoint
    from utils.checkpoints import save_checkpoint, load_checkpoint
    from utils.clearGPU import clearGPU
    from utils.ResumeTraining import ResumeTraining, LoadWeightsToModel
    from utils.saveParameters import saveParameters
    from testers import evaluate
else:
    from deep_rppg.utils.plotLossHistory import plotLossHistory
    from deep_rppg.utils.checkpoints import save_checkpoint
    from deep_rppg.utils.checkpoints import save_checkpoint, load_checkpoint
    from deep_rppg.utils.clearGPU import clearGPU
    from deep_rppg.utils.ResumeTraining import ResumeTraining, LoadWeightsToModel
    from deep_rppg.utils.saveParameters import saveParameters  
    from deep_rppg.testers import evaluate

def train_eval(args, model, train_ds, test_df, criterion, metric, optimizer, global_VERBOSE):
    """
    Function to train and evaluate the model using W&B, saving all logs in the cloud
    Args:
        args (argparse): Args with infor about the experiment
        model (torch.nn.Module): Model to be used
        train_ds (torch.utils.data.Dataset): Train dataset to be used
        test_df (pd.DataFrame): Test dataframe to be used
        criterion (torch.nn.Module): loss function to be used
        metric (torch.nn.Module): metric to be used in the test set
        optimizer (): Optimizer to be used
        global_VERBOSE(dict): All VERBOSE flags
    """    
    if args.dataset_percentage < 0: # If we are debugging log more frequentely
        log_loss_each_n_batches = 2
    else:
        log_loss_each_n_batches = 25
        
    save_and_log_each_epoch_n_epocs = {'signal_sample':1, 'model_checkpoint':1}
    
    VERBOSE = global_VERBOSE['train_eval']
    # check if resume training
    # best_model = wandb.restore('model-best.h5', run_path=args.save_path)    

    # LOAD WEIGHTS IF NEEDED (FINETUNING OR CROSS-DATASET)
    if args.load_these_weights == 'None':            
        if VERBOSE>0: print(f'=>[train_eval]: Prediction with {args.use_these_weights} weights')
        LoadWeightsToModel(args, model, optimizer, args.save_path, args.use_these_weights, args.epochs, args.fold, global_VERBOSE['LoadWeightsToModel'])
    else: #CROSS-DATASET
        # LOAD WEIGHTS IF NEEDED
        load_checkpoint(model, optimizer, args.load_these_weights)
        if VERBOSE>0: print(f'=>[load_these_weights] {args.load_these_weights} successfuly loaded in {args.network}')
    
    if not(args.predict_without_training):
        if VERBOSE>0: print('=>[train_eval]: Begin...')   
    
        if not(args.in_COLAB==True) and torch.cuda.is_available(): clearGPU(global_VERBOSE['clearGPU'])   
        # train_ds.plot_first_middle_last_sample(0)
        train_loader = DataLoader(train_ds, args.batch_size, drop_last=False, shuffle=False)
        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        if not(args.is_SWEEP): wandb.watch(model, criterion, log="all", log_freq=10)
    
        # RESUME TRAINING()
        if args.is_resume:
            is_Training_over, initial_epoch, example_ct, batch_ct, best_validation = ResumeTraining(args, model, optimizer, args.save_path, args.epochs, args.fold, global_VERBOSE['ResumeTraining'])
        else:
            if (not(args.is_SWEEP) and not(exists(join(abspath(args.save_path),str(args.fold),'weights')))): os.makedirs(join(abspath(args.save_path),str(args.fold),'weights'))
            is_Training_over, initial_epoch = False, 0
            total_batches, example_ct, batch_ct, best_validation  = len(train_loader) * args.epochs, 0, 0, np.inf
    
        if not(is_Training_over):
            
            for epoch in tqdm(range(initial_epoch, args.epochs)):  
                train_ds.ShuffleWindowsMetadata()
                if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128'] and args.lstm_state in ['stateful']:
                    model.reset_hidden() 
                                     
                for _, sample in enumerate(train_loader):
                    if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128'] and args.lstm_state in ['stateless']:
                        model.reset_hidden() 
                    model.train()           
                    
                    if args.network in ['PHYSNET','RTRPPG','RPPGNET','P64','PS1','PS2','PS3','PS4','PB1','PB2','UBS1','UB64','UB32','UB16','UB8','UB8_RGBYUV','UB4','UB2','UB64HR']:
                        loss = train_batch_PHYSNET(args, sample['x'], sample['y'], sample['t'], model, optimizer, criterion)              
                    elif args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
                        loss = train_batch_LSTMDF(args, sample['x'], sample['y'], sample['t'], model, optimizer, criterion)  
                        
                    example_ct +=  len(sample['x'])
                    batch_ct += 1  
                    
                    # Report metrics every 25th batch
                    if ((batch_ct + 1) % log_loss_each_n_batches) == 0:
                        train_log_loss(args, loss, example_ct, epoch)
        
    
                if args.fold in [0,1,2,3,4,-1]:
                    val_metric = evaluate(args, model, criterion, test_df, args.window, args.step_eval, epoch, example_ct, False, global_VERBOSE)
                else:
                    val_metric = loss   
    
                # Save best model if metric improves
                if (not(args.is_SWEEP) and (best_validation > val_metric.item())):
                    if VERBOSE>0: print(f'=>[train_eval]: IMPROVEMENT New best weigths found, from {best_validation:.5f} to {val_metric.item():.5f} saving model.')                    
                    best_validation = val_metric.item()   
                    checkpoint = {'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}
                    if not(exists(join(args.save_path,str(args.fold),'weights'))): os.makedirs(join(args.save_path,str(args.fold),'weights'))
                    save_checkpoint(checkpoint,join(args.save_path,str(args.fold),'weights','model_f{}_e{}.best.pth.tar'.format(str(args.fold),str(epoch).zfill(3))))

    
                # # Save everything you need once 1 epoch finish            
                save_and_log(args, model, optimizer, sample['x'], sample['y'], best_validation, example_ct, batch_ct, epoch, save_and_log_each_epoch_n_epocs,VERBOSE)

            if VERBOSE>0: print('=>[train_eval] TRAINING OVER.')     
            
        # Once training is over, predict
        
        if args.is_prediction:
            if VERBOSE>0: print(f'=>[train_eval]: Prediction with {args.use_these_weights} weights')
            LoadWeightsToModel(args, model, optimizer, args.save_path, args.use_these_weights, args.epochs, args.fold, global_VERBOSE['LoadWeightsToModel'])
            evaluate(args,model, criterion, test_df, args.window, 1, initial_epoch, example_ct, args.is_prediction, global_VERBOSE)  
            if VERBOSE>0: print(f'=>[train_eval]: PREDICTION OVER')
                
    else: # Only prediction
        
        if VERBOSE>0: print('=>[train_eval]: Only prediction') 
        evaluate(args, model, criterion, test_df, args.window, 1, 1, 1, True, global_VERBOSE)
        if VERBOSE>0: print(f'=>[train_eval]: PREDICTION OVER')

def save_and_log(args, model, optimizer, x, y, best_validation, example_ct, batch_ct, epoch, save_and_log_each_epoch_n_epocs, VERBOSE):
 
    
    # LOG ONE SIGNAL  
    if ((epoch + 1) % save_and_log_each_epoch_n_epocs['signal_sample']) == 0:    
        train_log_signal(args, x, y, model, example_ct, epoch)  
    # SAVE MODEL ARCHITECTURE
    if not(args.is_SWEEP):

        # UNCOMMENT TO SAVE MODEL ARCHITECTURE        
        # if epoch == 0: # Always save model architecture as onnx in the first epoch
        #     if args.in_COLAB==True:
        #         torch.onnx.export(model, x, f"{args.network}.onnx")
        #         wandb.save(f"{args.network}.onnx") 
        #         if VERBOSE>0: print(f'=>[train_eval] Saving {args.network}.onnx file')
        #     else:
        #         torch.onnx.export(model, x, join(args.save_path,f"{args.network}.onnx"))
        #         wandb.save(join(args.save_path,f"{args.network}.onnx"),base_path=args.save_path) 
        #         if VERBOSE>0: print(f'=>[train_eval] Saving {args.network}.onnx file')
        
        # SAVE CHECKPOINTS
        if ((epoch + 1) % save_and_log_each_epoch_n_epocs['model_checkpoint']) == 0:
            checkpoint = {'model_state_dict': model.state_dict(),# Model layers and weights
                          'optimizer_state_dict': optimizer.state_dict(),# Optimizer
                          'best_validation': best_validation,
                          'epoch': epoch,
                          'example_ct': example_ct,
                          'batch_ct': batch_ct,
                          }
            if not(exists(join(args.save_path,str(args.fold),'weights'))): os.makedirs(join(args.save_path,str(args.fold),'weights'))
            save_checkpoint(checkpoint,join(args.save_path,str(args.fold),'weights','model_f{}_e{}.pth.tar'.format(str(args.fold),str(epoch).zfill(3))))
        # if VERBOSE>0: print(f'=>[save_and_log] Saving checkpoint epoch {epoch}')


def train_batch_PHYSNET(args, x, y, t, model, optimizer, criterion):
    
    # Forward pass ➡
    y_hat, HR, _, _ = model(x)
    
    # Batch normalization?
    y = y-torch.mean(y,keepdim=True,dim=1)/torch.std(y,keepdim=True,dim=1)
    y_hat = y_hat-torch.mean(y_hat,keepdim=True,dim=1)/torch.std(y_hat,keepdim=True,dim=1)
            
    loss = criterion([y_hat, y, t, HR])
    
    #l1_strength = 0.01
    #l2_lambda = 0.001
    #l1_norm = sum(torch.linalg.matrix_norm(p, 1) for p in model.parameters())
    #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    #loss = loss + l2_lambda * l2_norm
    
    # Backward pass ⬅
    optimizer.zero_grad()
    #l1_loss = 0
    #for param in model.parameters():
        #l1_loss += torch.sum(torch.abs(param))
    #loss += l1_strength * l1_loss
    
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_batch_LSTMDF(args, x, y, t, model, optimizer, criterion):
    
    # Forward pass ➡
    y_hat,_,_,_ = model(x)
            
    loss = criterion([y_hat.squeeze(-1), y.squeeze(-1), t.squeeze(-1)])
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log_loss(args, loss, example_ct, epoch):
    # log to wandb
    wandb.log({"epoch": epoch, f'train/{args.loss}': float(loss.item())}, step=example_ct)
    
def train_log_signal(args, x, y, model, example_ct, epoch):
    with torch.no_grad():
        model.eval()
        if args.network in ['LSTMDFMTM125','LSTMDFMTM128','LSTMDFMTO128']:
            y_hat,_,_,_ = model(x)
            y_hat = y_hat.squeeze(-1)
            y = y.squeeze(-1)
            
        else:
            y_hat,_, _, _ = model(x)
            y = y-torch.mean(y,keepdim=True,dim=1)/torch.std(y,keepdim=True,dim=1)
            y_hat = y_hat-torch.mean(y_hat,keepdim=True,dim=1)/torch.std(y_hat,keepdim=True,dim=1)
        
        # Use matplotlib (it shows annoying warnings but Plotly was not working well)  
        try:           
            if args.in_COLAB:
                fig, ax = plt.subplots()
                ax.plot(y[0].detach().to('cpu').numpy()),
                ax.plot(y_hat[0].detach().to('cpu').numpy())
                ax.legend(['GT','Pred'])
                wandb.log({f'train_pred{epoch}': fig})#wandb.log({f"train_pred/Epoc {epoch}": fig})                
            else:
                fig, ax = plt.subplots()
                ax.plot(y[0].detach().to('cpu').numpy()),
                ax.plot(y_hat[0].detach().to('cpu').numpy())
                ax.legend(['GT','Pred'])
                wandb.log({f'train_pred{epoch}': fig})#wandb.log({f"train_pred/Epoc {epoch}": fig})
                plt.close('all')
        except:
            print(f'=>[train_log_signal] There was an error when trying to plot the first train in epoch {epoch}')