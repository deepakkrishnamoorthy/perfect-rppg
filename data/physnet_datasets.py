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
                 in_COLAB,load_dataset_path,
                 df:pd.DataFrame,
                 img_size:int=128,
                 window:int=128,
                 step:int=128,
                 HARD_ATTENTION:bool = False,
                 OnlySaveMetaData:bool=False,
                 db_name:str=None,
                 seed:int=10):
        """
        Args:
            args () : Experiment arguments, we will use: load_dataset_path and in_COLAB
            df (dataframe): Dataframe with the paths of all subjects
            img_size (int): Size of the square image
            window (int): window size
            step (int): window step
            HARD_ATTETION (bool): Wether use onlyl skin pixels as input
            is_RGBYUV(bool): Wheter use RGB and YUV channels as input
            OnlySaveMetaData(bool): Flag for only save metadata and not load files
            seed (int): seed to manage the np.random.RandomState to reproducible behaviour in ShuffleWindowsMetadata
            self.windowsFramesNames (List[List]): Paths withe the frames names per window
            self.GTindex (List[tuple]): Begind and End of GT per window
            self.db_name (string): database used

        """
        self.step = step
        self.df_subjects = df 
        self.img_size = img_size
        self.window = window
        self.is_HARD_ATTENTION = HARD_ATTENTION
        self.OnlySaveMetaData = OnlySaveMetaData
        self.db_name = db_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.path = abspath(load_dataset_path)
        self.in_COLAB = in_COLAB
        
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
            name = self.df_subjects['subject'].iloc[i]
            crt_npyfile = np.load(os.path.join(self.path,name,name+'.npy'))
            for i in range(0,np.size(crt_npyfile,0)-self.window+1,self.step):# For the entire video
                self._WindowsMetadata.append({'path':join(self.path,name),'name':name,'idx':(i,i+self.window)})

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
            Frames = self.getFullFramesFile(path,name)
            GroundTruth = self.getFullGTfile(path,name)
            TimeFile = self.getFullTimeFile(path,name,GroundTruth)
            # Take only current window
            frames = self.take_frames_crt_window(Frames,index,path,name)
            GT = torch.tensor(GroundTruth[index[0]:index[1]], dtype=torch.float32)
            time = torch.tensor(TimeFile[index[0]:index[1]], dtype=torch.float32)
            sample = {'x':frames.to(self.device), 'y':GT.to(self.device), 't':time.to(self.device), 'name':name, 'index': index}
        return sample    

    def getFullFramesFile(self,path:str,name:str)-> np.array:
        frames = np.load(os.path.join(path,name+'.npy'))
            
        return frames
    
    def getFullGTfile(self,path:str,name:str)->np.array:
        if self.db_name in ['VIPL','MMSE','COHFACE','UBFC','BIGECGF','MRL']:
            return np.loadtxt(os.path.join(path,name+'_gt.txt'))
        elif self.db_name in ['ECGF']:
            return np.load(os.path.join(path,name+'_gt.npy'))
            

    def getFullTimeFile(self,path:str,name:str,GroundTruth)->np.array:
        if self.db_name in 'VIPL':# VIPL gives its own timestamp file
            return np.loadtxt(os.path.join(path,name+'_timestamp.txt'))
        elif self.db_name in ['ECGF','BIGECGF','UBFC','BIGUBFC','MRL']: # fps in ECGF after pre-processing 30Hz
            timeTrace = np.arange(0,(1./30.)*len(GroundTruth),(1./30.)) 
            return timeTrace   
        elif self.db_name in ['MMSE']:# fps in MMSE after pre-processing 25Hz
            timeTrace = np.arange(0,(1./25.)*len(GroundTruth),(1./25.)) 
            return timeTrace
        elif self.db_name in ['COHFACE']:# fps in COHFACE 20Hz
            timeTrace = np.arange(0,(1./20.)*len(GroundTruth),(1./20.)) 
            return timeTrace

    # FUNCTION TO TAKE ONLY THE FRAMES OF THE CURRENT WINDOW AND RETURN A TENSOR OF IT
    def take_frames_crt_window(self,Frames,idx,path,name):
        
        frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
            
        if self.is_HARD_ATTENTION: BoolSkinMask = np.load(os.path.join(path,name+'_skinmask.npy'))
        # Load all frames in current window
        for j,i in enumerate(range(idx[0],idx[1])):
            frame = np.array(Frames[i,:,:,:].copy(),dtype=np.float32)/255.0 
            #cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO);cv2.imshow('frame',frame)            
            # If it is hard_attetion, take only skin values, set 0 others
            if self.is_HARD_ATTENTION: 
                frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, borderType=cv2.BORDER_REPLICATE) 
                Floatmask = cv2.copyMakeBorder(np.array(BoolSkinMask[i,:,:],dtype=np.float32), 5, 5, 5, 5, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
                frame = frame*cv2.merge((Floatmask,Floatmask,Floatmask))
                image_mean = cv2.mean(frame, mask=np.array(Floatmask,dtype=np.uint8))
                frame[~np.array(Floatmask,dtype=np.bool),:] = [image_mean[0],image_mean[1],image_mean[2]]
                frame = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)
                
            # If current image size is different than img_size, resize
            if self.img_size != frame.shape[-2]:
                frame = cv2.resize(frame,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)

            # Swap image channels for pytorch
            frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
            frames[:,j,:,:]=frame 
        #plt.figure(),plt.imshow(np.array(frames[:,-1,:,:].permute(1,2,0)*255,dtype=np.uint8))
        return frames

    def plot_first_middle_last_sample(self,idx):
        sample = self[idx]
        x = sample['x'].to('cpu')
        y = sample['y'].to('cpu')
        name = sample['name']
        index = sample['index']
        print(f'Sujbect {name}, window {index}')
        fig = plt.figure()
        fig.suptitle(name)
        relative_first = 0 # relative position of first frame
        relative_Last = self.window-1 # relative position of end frame
        relative_middle = int((relative_Last-relative_first)/2)

        absolute_first = index[0] # absolute position of first frame
        absolute_middle = absolute_first+relative_middle# absolute position of middle frame
        absolute_Last = index[-1]-1 # absolute position of last frame
        
        IDX = 0 # index current figure
        for i,j in zip([relative_first,relative_middle,relative_Last],[absolute_first,absolute_middle,absolute_Last]):
            ax = plt.subplot(1, 3, IDX+1)
            plt.tight_layout()
            ax.set_title('frame {}({}),GT={:0.4f}'.format(j,i,y[i]),fontsize=10)
            ax.axis('off')
            crt_frame = x[:,i,:,:]#take current frame from tensor
            crt_frame = np.array(crt_frame.permute(1,2,0))
            crt_frame_rgb = cv2.cvtColor(crt_frame, cv2.COLOR_BGR2RGB)
            plt.imshow(crt_frame_rgb)
            IDX = IDX + 1 
            if i == relative_Last:
                plt.show()
#%%
# CUSTOM DATASET TO LOAD DATASET INDEPENDTHLY?
class SubjectIndependentTestDataset(torch.utils.data.Dataset):
    """This dataset takes one dataframe  with one single subject, it divide it with a sliding window
    with ALWAYS A STEP=1 and do the model evaluation. This dataset should be use after using getSubjectIndependentTrainValDataFrames.
    At the end of one subject predicition, it uses overlap-add procedure to have the final output.
    """
 
    def __init__(self, in_COLAB:bool=False,
                 load_dataset_path:str=r'content/data',
                 subject_df:pd.DataFrame=None,
                 img_size:int=128,
                 window:int=128,
                 step:int=1,
                 HARD_ATTENTION:bool = False,
                 db_name:str=None):
        """
        Args:
            in_COLAB (bool): Flag to change load_dataset_path to r'content/data'
            load_dataset_path(str): Path where the data is located (only used if in_COLAB=False)
            subject_df (pd.DataFrame): DataFrame of one the one single subject
            window (int): window size
            step (int): sliding window step
            HARD_ATTENTION (bool): Wether use onlyl skin pixels as input
            db_name (str): name of the database in use
            is_RGBYUV(bool): Wheter use RGB and YUV channels as input
            self.windowsFramesNames (List[List]): Paths withe the frames names per window
            self.GTindex (List[tuple]): Begind and End of GT per window
            self.name (string): subject name
            self.architecture (string): name of the architecture to use

        """
        self.in_COLAB = in_COLAB
        self.load_path = abspath(load_dataset_path)
        self.subject_df = subject_df
        self.img_size = img_size
        self.step = step 
        self.window = window
        self.db_name = db_name
        self.name = self.subject_df['subject']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_HARD_ATTENTION = HARD_ATTENTION
   
        # If we are in COLAB the data will be always in the same path
        if self.in_COLAB==True:
            self.load_path = r'/content/data'

        self.load_path = os.path.join(self.load_path,self.name)
        # LOAD ALL FRAMES SORTED IN self.root
        self.frames = np.load(os.path.join(self.load_path,self.name+'.npy'))
        self.x_file = self.frames # Only to work with LSTMDF datasets
        if self.is_HARD_ATTENTION: self.BoolSkinMask = np.load(os.path.join(self.load_path,self.name+'_skinmask.npy'))

        self.Frames = np.zeros((self.frames.shape[0],self.img_size,self.img_size,3),dtype=np.float32)
        
        
        for i in range(0,self.frames.shape[0]):
            frame = np.array(self.frames[i,:,:,:].copy(),dtype=np.float32)/255.0            
            # If it is hard_attetion, take only skin values, set 0 others
            if self.is_HARD_ATTENTION: 
                frame = cv2.copyMakeBorder(frame, 5, 5, 5, 5, borderType=cv2.BORDER_REPLICATE) 
                Floatmask = cv2.copyMakeBorder(np.array(self.BoolSkinMask[i,:,:],dtype=np.float32), 5, 5, 5, 5, borderType=cv2.BORDER_CONSTANT,value=[0,0,0])           
                frame = frame*cv2.merge((Floatmask,Floatmask,Floatmask))
                image_mean = cv2.mean(frame, mask=np.array(Floatmask,dtype=np.uint8))
                frame[~np.array(Floatmask,dtype=np.bool),:] = [image_mean[0],image_mean[1],image_mean[2]]
                frame = cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT)

            # If current image size is different than img_size, resize
            if self.img_size != frame.shape[-2]:
                frame = cv2.resize(frame,(self.img_size,self.img_size),interpolation=cv2.INTER_AREA)
            self.Frames[i,:,:,:] = frame
        
        # LOAD GT FILE
        self.y_file = self.getFullGTfile()        
        # LOAD TIME FILE IF EXIST
        self.t_file = self.getFulltimeFile()
        # Get windows index
        self.windows = self.getWindows()
            
    # RETURN NUMBER OF WINDOWS
    def __len__(self):
            return len(self.windows)

    # RETURN THE [i] FILE
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Take only current window from big tensor
        frames = self.take_frames_crt_window(self.windows[idx])
        GT = torch.tensor(self.y_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        time = torch.tensor(self.t_file[self.windows[idx][0]:self.windows[idx][1]], dtype=torch.float32)
        sample = {'x':frames.to(self.device), 'y':GT.to(self.device), 't':time.to(self.device)}
        return sample
    

    # FUNCTION TO GET THE WINDOWS INDEX
    def getWindows(self):
        windows = []
        for i in range(0,np.size(self.Frames,0)-self.window+1,self.step):
            windows.append((i,i+self.window))
        return windows

    # FUNCTION TO TAKE ONLY THE FRAMES OF THE CURRENT WINDOW AND RETURN A TENSOR OF IT
    def take_frames_crt_window(self,idx):
        frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
        
        # Load all frames in current window
        for j,i in enumerate(range(idx[0],idx[1])):
            frame = self.Frames[i,:,:,:]
            frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
            frames[:,j,:,:] = frame 
        #plt.imshow(frames[:,0,:,:].permute(1,2,0)*255)
        return frames

    # def take_frames_crt_window(self,idx):
    #     frames = torch.zeros((3,self.window,self.img_size,self.img_size)) # list with all frames {3,T,128,128}
    #     # Load all frames in current window
    #     for j,i in enumerate(range(idx[0],idx[1])):
    #         frame = self.Frames[i,:,:,:].copy()
    #         # If it is hard_attetion, take only skin values, set 0 others
    #         if self.is_HARD_ATTENTION:  
    #             lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #             gray_img = lab_image[:,:,0]/255.0
    #             x = np.zeros((1, frame.shape[-2], frame.shape[-2], 1),np.float)
    #             x[0,:,:,0] = gray_img
    #             if self.in_COLAB: 
    #                 y_pred = self.skin_model.predict(x)      
    #             else:
    #                 with tf.device('/cpu:0'):
    #                     y_pred = self.skin_model.predict(x) 
    #             mask = np.array(255*y_pred[0,:,:,0],dtype=np.uint8)
    #             frame = cv2.bitwise_and(frame,frame,mask = mask)
    #         # If current image size is different than img_size, resize
    #         if self.img_size != frame.shape[-2]:
    #             frame = cv2.resize(frame,(self.img_size,self.img_size),interpolation=cv2.INTER_CUBIC)
    #         # Normalize between 0-1 and turn to tensor
    #         frame = frame/255.0            

    #         # Swap image channels for pytorch
    #         frame = torch.tensor(frame, dtype=torch.float32).permute(2,0,1)#In pythorch channels must be in position 0, cv2 has channels in 2
    #         frames[:,j,:,:]=frame 
    #     #plt.figure(),plt.imshow(np.array(frames[:,-1,:,:].permute(1,2,0)*255,dtype=np.uint8))
    #     return frames
    
    def plot_first_middle_last_sample(self,idx):
        print('Sujbect {}, window {}'.format(self.name,idx))
        fig = plt.figure()
        fig.suptitle(self.name)
        sample = self[idx]
        x = sample['x'].detach().to('cpu')
        y = sample['y'].detach().to('cpu')
        relative_first = 0 # relative position of first frame
        relative_Last = self.window-1 # relative position of end frame
        relative_middle = int((relative_Last-relative_first)/2)

        absolute_first = self.windows[idx][0] # absolute position of first frame
        absolute_middle = absolute_first+relative_middle# absolute position of middle frame
        absolute_Last = self.windows[idx][-1]-1 # absolute position of last frame
        
        IDX = 0 # index current figure
        for i,j in zip([relative_first,relative_middle,relative_Last],[absolute_first,absolute_middle,absolute_Last]):
            ax = plt.subplot(1, 3, IDX+1)
            plt.tight_layout()
            ax.set_title('frame {}({}),GT={:0.4f}'.format(j,i,y[i]),fontsize=10)
            ax.axis('off')
            crt_frame = x[:,i,:,:]#take current frame from tensor
            crt_frame = np.array(crt_frame.permute(1,2,0))
            crt_frame_rgb = cv2.cvtColor(crt_frame, cv2.COLOR_BGR2RGB)
            plt.imshow(crt_frame_rgb)
            IDX = IDX + 1 
            if i == relative_Last:
                plt.show()

    def getFullGTfile(self):
        GTname = glob.glob(os.path.join(self.load_path,'*_gt*'))[-1]
        if self.db_name in ['VIPL','UBFC','BIGECGF','MMSE','COHFACE','MRL']:
            return np.loadtxt(GTname)
        elif self.db_name in ['ECGF','VIPL_ECGF','BIGECGF']:
            return np.load(GTname)
        #elif self.db_name in ['MRL']:
            #print(path, name)
            #gtfile = np.loadtxt(os.path.join(path,name+'_gt.txt'))
            #return np.loadtxt(os.path.join(path,name+'_gt.txt'))
    
    def getFulltimeFile(self):
        if self.db_name == 'VIPL':
            timename = glob.glob(os.path.join(self.load_path,'*_timestamp*'))[-1]
            return np.loadtxt(timename)
        elif self.db_name in ['MMSE','VIPL_MMSE']:
            timeTrace = np.arange(0,(1./25.)*self.Frames.shape[0],(1./25.)) 
            return timeTrace
        elif self.db_name in ['ECGF','VIPL_ECGF','UBFC','BIGECGF','MRL']:
            timeTrace = np.arange(0,(1./30.)*self.Frames.shape[0],(1./30.)) 
            return timeTrace             
        elif self.db_name in ['COHFACE']:
            timeTrace = np.arange(0,(1./20.)*self.Frames.shape[0],(1./20.)) 
            return timeTrace  
