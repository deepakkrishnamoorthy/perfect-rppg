import torch
import numpy as np
import os
import sys
from os.path import join, abspath
import random
from natsort import natsorted
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as T
import torchvision
from datetime import datetime
import pandas as pd
import glob
#from tensorflow.keras.models import load_model
#from tensorflow.keras import backend as K
#import tensorflow as tf
from time import perf_counter
from scipy.io import loadmat

#def asymmetric_loss(y_true,y_pred):
#	alpha=0.30
#	return K.mean(K.abs(y_pred - y_true)*(alpha*y_true+(1-alpha)*(1-y_true)), axis=-1)

# CUSTOM DATASET TO LOAD DATASET INDEPENDTHLY?
class TrainDataset(torch.utils.data.Dataset):
    """This dataset takes one dataframe with subjects independent folders, it divide all subjects by sliding window and
    with the ShuffleWindowsMetadata function it generates random batches with info of all subjects. 
    This dataset only should be use after using getSubjectIndependentTrainValDataFrames function.
    """

    def __init__(self, is_reproducible,
                 in_COLAB,
                 df:pd.DataFrame,
                 window:int=128,
                 step:int=128,
                 OnlySaveMetaData:bool=False,
                 db_name:str=None,
                 many_to:str='MTM',
                 seed:int=10):
        """
        Args:
            args () : Experiment arguments, we will use: load_dataset_path and in_COLAB
            df (dataframe): Dataframe with the paths of all subjects
            window (int): window size
            step (int): window step
            OnlySaveMetaData(bool): Flag for only save metadata and not load files
            seed (int): seed to manage the np.random.RandomState to reproducible behaviour in ShuffleWindowsMetadata
            self.windowsFramesNames (List[List]): Paths withe the frames names per window
            self.GTindex (List[tuple]): Begind and End of GT per window
            self.db_name (string): database used

        """
        self.step = step
        self.df_subjects = df 
        self.window = window
        self.OnlySaveMetaData = OnlySaveMetaData
        self.db_name = db_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.in_COLAB = in_COLAB
        self.many_to = many_to
        
        if is_reproducible:     
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(hash(datetime.now())% 2**32 - 1)      
       
        # If we are in COLAB the data will be always in the same path
        if self.in_COLAB==True:
            self.path = r'/content/data'
        # Create list with paths and index for each window 
        self.CreateWindowsMetadata()

    def CreateWindowsMetadata(self):
        '''Function to create dictionary with Metadata of windows'''
        self._WindowsMetadata = []
        for i in range(0,len(self.df_subjects)):# For all subjects
            path = self.df_subjects['path'].iloc[i]
            name = self.df_subjects['subject'].iloc[i]
            file_length = np.size(loadmat(join(path,name))['GT'],1)
            if self.many_to in ['MTM']:
                for i in range(0,file_length-self.window+1,self.step):# For the entire video
                    self._WindowsMetadata.append({'path':path,'name':name,'idx':(i,i+self.window)})
            elif self.many_to in ['MTO']:#When MTO we want to stop on the penultimate value
                for i in range(0,file_length-self.window,self.step):# For the entire video
                    self._WindowsMetadata.append({'path':path,'name':name,'idx':(i,i+self.window)})                

    def ShuffleWindowsMetadata(self):
        '''Function to reshuffle the windows metadata for trainings'''
        self.rng.shuffle(self._WindowsMetadata)            
        
    @property
    def getWindowsMetadata(self):
        '''Function to return WindowsMetadata'''
        return self._WindowsMetadata
        
    # RETURN NUMBER OF WINDOWS
    def __len__(self)->int:
        return len(self._WindowsMetadata)
    
    # RETURN THE [i] FILE
    def __getitem__(self, idx):
        # Take metadata
        path = self._WindowsMetadata[idx]['path']
        name = self._WindowsMetadata[idx]['name']
        index = self._WindowsMetadata[idx]['idx']
        
        if self.OnlySaveMetaData:
            # If we onlyl need to save metadata there is no need of loading the files
            sample = {'name':name, 'index': index}
        else:
            # Load files  
            x_file, y_file, t_file = self.get_x_y_t_files(path,name)# Get subject data for the full video
            # Take only current window
            x = torch.tensor(x_file[index[0]:index[1]], dtype=torch.float32).unsqueeze(-1)
            t = torch.tensor(t_file[index[0]:index[1]], dtype=torch.float32).unsqueeze(-1)
            if self.many_to in ['MTM']:#
                y = torch.tensor(y_file[index[0]:index[1]], dtype=torch.float32).unsqueeze(-1)
            elif self.many_to in ['MTO']:#In MTO we take 128 inputs to predict the next value
                y = torch.tensor(y_file[index[1]], dtype=torch.float32).unsqueeze(-1)
            sample = {'x':x.to(self.device), 'y':y.to(self.device), 't':t.to(self.device), 'name':name, 'index': index}
        return sample    

    def get_x_y_t_files(self,path:str,name:str)-> np.array:
        # Function to get the input rPPG, Ground truth and time
        mat_file = loadmat(join(path,name))
        x_file = mat_file['Pred'][0]
        y_file = mat_file['GT'][0]
        t_file = mat_file['timeTrace'][0]
        # import matplotlib.pyplot as plt
        # plt.figure(),plt.plot(t_file,y_file,'r');plt.plot(t_file,x_file,'b')
            
        return x_file, y_file, t_file

#%%
# CUSTOM DATASET TO LOAD DATASET INDEPENDTHLY?
class SubjectIndependentTestDataset(torch.utils.data.Dataset):
    """This dataset takes one dataframe  with one single subject, it divide it with a sliding window
    with ALWAYS A STEP=1 and do the model evaluation. This dataset should be use after using getSubjectIndependentTrainValDataFrames.
    At the end of one subject predicition, it uses overlap-add procedure to have the final output.
    """
 
    def __init__(self, in_COLAB:bool=False,
                 subject_df:pd.DataFrame=None,
                 window:int=128,
                 step:int=1,
                 db_name:str=None,
                 many_to:str='MTM'):
        """
        Args:
            in_COLAB (bool): Flag to change load_dataset_path to r'content/data'
            subject_df (pd.DataFrame): DataFrame of one the one single subject
            window (int): window size
            step (int): sliding window step
            db_name (str): name of the database in use
            self.windowsFramesNames (List[List]): Paths withe the frames names per window
            self.GTindex (List[tuple]): Begind and End of GT per window
            self.name (string): subject name
            self.architecture (string): name of the architecture to use

        """
        self.in_COLAB = in_COLAB
        self.path = subject_df['path']
        self.name = subject_df['subject']
        self.step = step 
        self.window = window
        self.db_name = db_name
        self.many_to = many_to
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
        # If we are in COLAB the data will be always in the same path
        # if self.in_COLAB==True:
        #     self.load_path = r'/content/data'

        # Load complete subject data        
        self.x_file, self.y_file, self.t_file = self.get_x_y_t_files(self.path,self.name)# Get subject data for the full video

        # Get windows index
        self.windows = self.getWindows()

    def get_x_y_t_files(self,path:str,name:str)-> np.array:
        # Function to get the input rPPG, Ground truth and time
        mat_file = loadmat(join(path,name))
        x_file = mat_file['Pred'][0]
        y_file = mat_file['GT'][0]
        t_file = mat_file['timeTrace'][0]
        # import matplotlib.pyplot as plt
        # plt.figure(),plt.plot(t_file,y_file,'r');plt.plot(t_file,x_file,'b')
            
        return x_file, y_file, t_file

    # FUNCTION TO GET THE WINDOWS INDEX
    def getWindows(self):
        windows = []
        if self.many_to in ['MTM']:
            for i in range(0,np.size(self.x_file,0)-self.window+1,self.step):
                windows.append((i,i+self.window))
        elif self.many_to in ['MTO']:#When MTO we want to stop on the penultimate value
            for i in range(0,np.size(self.x_file,0)-self.window,self.step):
                windows.append((i,i+self.window))

        return windows            
    # RETURN NUMBER OF WINDOWS
    def __len__(self):
            return len(self.windows)

    # RETURN THE [i] FILE
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Take only current window from big tensor
        x = torch.tensor(self.x_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32).unsqueeze(-1)
        t = torch.tensor(self.t_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32).unsqueeze(-1)
        if self.many_to in ['MTM']:#
            y = torch.tensor(self.y_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32).unsqueeze(-1)
        elif self.many_to in ['MTO']:#In MTO we take 128 inputs to predict the next value
            y = torch.tensor(self.y_file[self.windows[idx][1]], dtype=torch.float32).unsqueeze(-1)        
        
        sample = {'x':x.to(self.device), 'y':y.to(self.device), 't':t.to(self.device)}
        return sample   


