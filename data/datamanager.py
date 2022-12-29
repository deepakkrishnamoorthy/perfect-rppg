try:
    if 'google.colab' in str(get_ipython()):
        COLAB = True
    else:
        COLAB = False
except:
    COLAB = False

from abc import abstractmethod
import pandas as pd
import zipfile
import os
import numpy as np
from natsort import natsorted
import random
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.utils import shuffle
import shutil
from os.path import join, abspath, exists
import collections
from torch.utils.data import DataLoader
from datetime import datetime 
import scipy.io
from distutils.dir_util import copy_tree

if not(COLAB):
    from data.physnet_datasets import TrainDataset as PhysNetTrainDataset
else:
    from deep_rppg.data.physnet_datasets import TrainDataset as PhysNetTrainDataset

# %% First DataManager attempt with LSTM-DF
def DataManager(args, root:str, db_name:str, is_reproducible:bool=True, seed:int=10, percentage:float=1, VERBOSE:int=1, is_COLAB:bool=False):
    """
    Function to manage the datasets and the amount of data to take
    Args:
        seed (): seed to set a random number generator to manage reproducibility
        root (str) : Path where the dataset is located
        db_name(str) : name of the database
        percentage(float): amount of data to take of the dataset
        VERBOSE (int): the higher the more info about the process will be printed
        is_COLAB (bool): Flag indicating that we are working in Google Colab
    """
    root = abspath(root)
    if is_reproducible:     
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(hash(datetime.now())% 2**32 - 1)             

    if args.is_from_rPPG: # Managing rPPG signals as input
        """
        Here, we are going to train from predictions given by another model, that means that the input expected is 
        5 folders called 0,1,2,3, and 4 with the rPPG signals.
        """
    
        print('[DataManager] Training from rPPG signals instead of video-frames') 
        assert('0' in os.listdir(args.load_dataset_path)),'[DataManager][ERROR] Input Folder 0 does not exist'
        assert('1' in os.listdir(args.load_dataset_path)),'[DataManager][ERROR] Input Folder 1 does not exist'
        assert('2' in os.listdir(args.load_dataset_path)),'[DataManager][ERROR] Input Folder 2 does not exist'
        assert('3' in os.listdir(args.load_dataset_path)),'[DataManager][ERROR] Input Folder 3 does not exist'
        assert('4' in os.listdir(args.load_dataset_path)),'[DataManager][ERROR] Input Folder 4 does not exist'
        
       
        DataPath = args.load_dataset_path
        
        if is_COLAB == True:  #  is_COLAB = True      
            Folder_Drive_name = r'/content/data' # Folder_Drive_name = args.save_path
            if not(os.path.exists(Folder_Drive_name)):
                os.mkdir(Folder_Drive_name)
                copy_tree(DataPath, Folder_Drive_name)
            args.load_dataset_path = r'/content/data'
        
    else:# Managing video-frames as input
    
        ###
        # In Google Colab, copy and paste the files to be used    
        if is_COLAB==True: # True:is_COLAB   
            # Get name file
            if '_' in db_name: # If cross dataset then choose the second one
                db_name = db_name.split('_')[1]
            if VERBOSE>0: print(f'=>[DataManager] Working in {db_name}{args.channels}{args.img_size} dataset, input {args.img_size}') 
            filename = os.path.join(abspath(root),f'{db_name}{args.channels}{args.img_size}_npy.zip')  
            print(f'=>[DataManager] Zip file: {filename}')

            Folder_Drive_name = r'/content/data' # r'E:\results\VIPL\test_dataset' #r'/content/data'
            if not(os.path.exists(Folder_Drive_name)):
                os.mkdir(Folder_Drive_name)
                archive = zipfile.ZipFile(filename, 'r') 
                archive.extractall(path=Folder_Drive_name)   
            else:
                print(f'[DataManager] {Folder_Drive_name} already exist, assuming that the data is complete')
            # # Take only nonempty folders
            # subjects_names =  [os.path.dirname(x) for x in archive.namelist()]
            # subjects_names = [item for item, count in collections.Counter(subjects_names).items() if count > 1]
            # subjects_names = natsorted(subjects_names)
            # df_subjects = pd.DataFrame({'subject':subjects_names})
          
            # Folder_Drive_name = r'/content/data' # r'E:\results\VIPL\test_dataset' #r'/content/data'
            # if not(os.path.exists(Folder_Drive_name)): os.mkdir(Folder_Drive_name)
            # for idx,subj in df_subjects.iterrows():
            #     subject_to_uncompress = subj['subject']
            #     if os.path.exists(join(Folder_Drive_name,subject_to_uncompress)):
            #         # If the folder already exist check that there are not missing files and then go the next subject 
            #         if db_name=='VIPL': # Check number of files
            #             if args.hard_attention == True:
            #                 n_files = 4
            #             else:
            #                 n_files = 3
                            
            #             if len([f for f in os.listdir(join(Folder_Drive_name,subject_to_uncompress))if os.path.isfile(join(join(Folder_Drive_name,subject_to_uncompress), f))])<n_files:
            #                 print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(df_subjects)}')           
            #                 archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
            #             else:# The subject folder is complete, therefore go to the next one
            #                     continue  
                            
            #         elif db_name in ['ECGF','MMSE','UBFC','BIGUBFC','BIGECGF']: # Check number of files
            #             # label file
            #             label_f_name = [x for x in archive.namelist() if x.endswith('labels.pickle')]
            #             #[print(x) for x in archive.namelist()]
            #             if len(label_f_name)>0:
            #                print(label_f_name[0])
            #                #archive.extractall(path=Folder_Drive_name,members=label_f_name[0])
            #                archive.extractall(path=Folder_Drive_name,members='subject9/subject9_gt.txt')
            #             if len([f for f in os.listdir(join(Folder_Drive_name,subject_to_uncompress))if os.path.isfile(join(join(Folder_Drive_name,subject_to_uncompress), f))])<2:
            #                 print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(df_subjects)}')           
            #                 archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
            #             else:# The subject folder is complete, therefore go to the next one
            #                     continue 
    
            #     else:
            #         print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(df_subjects)}')           
            #         archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
            # print(f'=>[DataManager] {len(df_subjects)} subjects loaded succesfully in {Folder_Drive_name}') 
            root = r'/content/data'

        ###
        # Take only nonempty folders        
        subjects_names_list = [join(root,x) for x in os.listdir(root) if os.path.isdir(join(root,x))] # take folders
        subjects_names_list = [folder for folder in subjects_names_list if len(os.listdir(folder))>0] # Ignore empty folders
        subjects_names_list = [folder.split(os.path.sep)[-1] for folder in subjects_names_list] # Take only subject's name
        subjects_names_list = natsorted(subjects_names_list)
    
        ###
        # Create scenarios for stratification        
        if db_name in ['COHFACE']:
            # In Cohface the last number is the scenario
            scenario_list = [x[-1] for x in subjects_names_list]           
            DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list})

        elif db_name=='VIPL':
            # In VIPL-HR, v is the scenario and s is the source (3 cameras)
            scenario_list = [] 
            source_list = [] 
            for name in subjects_names_list:
                scenario_list.append(int(name[name.find('v')+1]))
                source_list.append(int(name[name.find('s')+1]))
            
            DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list,'source':source_list})
        
        elif db_name in ['ECGF','VIPL_ECGF'] :
            scenario_list = [] 
            for name in subjects_names_list:
                scenario_list.append(int(name.split('_')[1]))
            
            DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list})  
            
        elif db_name in ['MMSE','UBFC','BIGECGF','MRL']:
            # scenario_list = [] 
            # for name in subjects_names_list:
            #     scenario_list.append(int(name.split('_')[1]))
            
            # DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list})
            
            # Find labels pickle file (pandas Dataframe)
            labels_file_name = [x for x in os.listdir(root) if x.endswith('labels.pickle')][0]
            df_labels = pd.read_pickle(join(root,labels_file_name))

            #Concatenate the subjects with their levels only if they are the same subjects in the same order            
            if np.all(df_labels['subject'].to_list()==subjects_names_list):
                scenario_list = df_labels['label'].to_list()                
                DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list})
            else:
                print('[DataManager]=>Error creating scenario labels in DataPath, scenario_list and def_labels do not have the same subjects')
 
        
        ###
        # Split data if necessary
        if percentage == 1:
            if VERBOSE>0: print(f'=>[DataManager] Taking 100% of {db_name}')
        elif percentage < 0:
            DataPath = DataPath[0:int(-1*percentage)]
            if VERBOSE>0: print(f'=>[DataManager] WARNING! Taking only {int(-1*percentage)} subject(s) of {db_name} (for DEBUG ONLY)')
        else: #     
            if VERBOSE>0: print(f'=>[DataManager] Taking {percentage*100}% of {db_name}')
            DataPath, _ = train_test_split(DataPath,train_size=percentage,random_state=rng,stratify=DataPath['scenario'])
            DataPath = DataPath.reset_index(drop=True)

    return DataPath

# %% First DataManager attempt with LSTM-DF
# Use this version of DataMager only if you are planing to continue working with LSTMDF128 taking the rPPG files from a single zip file
def DataManager2(args, root:str, db_name:str, is_reproducible:bool=True, seed:int=10, percentage:float=1, VERBOSE:int=1, is_COLAB:bool=False):
    """
    Function to manage the datasets and the amount of data to take
    Args:
        seed (): seed to set a random number generator to manage reproducibility
        root (str) : Path where the dataset is located
        db_name(str) : name of the database
        percentage(float): amount of data to take of the dataset
        VERBOSE (int): the higher the more info about the process will be printed
        is_COLAB (bool): Flag indicating that we are working in Google Colab
    """
    if is_reproducible:     
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(hash(datetime.now())% 2**32 - 1)             

    if args.network in ['LSTMDF128']: # Managing Input file to LSTMDF128 from predicted values of RTrPPG
        print('[DataManager] The LSTMDF128 implementation works with rPPG.mat signals predicted from previous models like RTrPPG')
        if db_name in 'VIPL':
            if VERBOSE>0: print('=>[DataManager] Working in VIPL')   
            filename = os.path.join(abspath(root),'VIPLRPPG_RTRPPG.zip')

        print(f'=>[DataManager] Zip file: {filename}')
        archive = zipfile.ZipFile(filename, 'r')
        subjects_names_list = natsorted(archive.namelist())
        
    else:# Others networks different than LSTMDF128
    
        # Get dataframe with subjects names from the compressed file .zip
        if db_name in 'VIPL':
            if VERBOSE>0: print(f'=>[DataManager] Working in VIPL{args.channels}{args.img_size} dataset input {args.img_size}')   
            filename = os.path.join(abspath(root),f'VIPL{args.channels}{args.img_size}_npy.zip')
    
        elif db_name in 'MMSE':
            if VERBOSE>0: print(f'=>[DataManager] Working in MMSE{args.channels}{args.img_size} dataset input {args.img_size}')
            filename = os.path.join(abspath(root),f'MMSE{args.channels}{args.img_size}_npy.zip')
        elif db_name in ['ECGF','VIPL_ECGF']:
            if VERBOSE>0: print(f'=>[DataManager] Working in ECGFitness{args.channels}{args.img_size} dataset {args.img_size}')
            filename = os.path.join(abspath(root),f'ECGF{args.channels}{args.img_size}_npy.zip')      
    
        print(f'=>[DataManager] Zip file: {filename}')
        archive = zipfile.ZipFile(filename, 'r')

        # Take only nonempty folders
        subjects_names_list =  [os.path.dirname(x) for x in archive.namelist()]
        subjects_names_list = [item for item, count in collections.Counter(subjects_names_list).items() if count > 1]
        subjects_names_list = natsorted(subjects_names_list)
    
    # Create scenarios for stratification
    if db_name=='VIPL':
        scenario_list = [] 
        source_list = [] 
        for name in subjects_names_list:
            scenario_list.append(int(name[name.find('v')+1]))
            source_list.append(int(name[name.find('s')+1]))
        
        DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list,'source':source_list})
    elif db_name=='MMSE':
        # TODO: create one "scenario" for MMSE
        pass
    elif db_name in ['ECGF','VIPL_ECGF','MRL'] :
        scenario_list = [] 
        for name in subjects_names_list:
            scenario_list.append(int(name.split('_')[1]))
        
        DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list})  
    
    # Split data if necessary
    if percentage == 1:
        if VERBOSE>0: print(f'=>[DataManager] Taking 100% of {db_name}')
    elif percentage < 0:
        DataPath = DataPath[0:int(-1*percentage)]
        if VERBOSE>0: print(f'=>[DataManager] WARNING! Taking only {int(-1*percentage)} subject(s) of {db_name} (for DEBUG ONLY)')
    else: #     
        if VERBOSE>0: print(f'=>[DataManager] Taking {percentage*100}% of {db_name}')
        DataPath, _ = train_test_split(DataPath,train_size=percentage,random_state=rng,stratify=DataPath['scenario'])
        DataPath = DataPath.reset_index(drop=True)

    # In Google Colab, copy and paste the files to be used    
    if is_COLAB==True: # True:is_COLAB   
        Folder_Drive_name = r'/content/data' # r'E:\results\VIPL\test_dataset' #r'/content/data'
        if not(os.path.exists(Folder_Drive_name)): os.mkdir(Folder_Drive_name)
        for idx,subj in DataPath.iterrows():
            subject_to_uncompress = subj['subject']
            if os.path.exists(join(Folder_Drive_name,subject_to_uncompress)):
                # If the folder already exist check that there are not missing files and then go the next subject 
                if db_name=='VIPL': # Check number of files
                    if args.hard_attention == True:
                        n_files = 4
                    else:
                        n_files = 3
                        
                    if len([f for f in os.listdir(join(Folder_Drive_name,subject_to_uncompress))if os.path.isfile(join(join(Folder_Drive_name,subject_to_uncompress), f))])<n_files:
                        print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(DataPath)}')           
                        archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
                    else:# The subject folder is complete, therefore go to the next one
                            continue  
                        
                elif db_name=='ECGF' or db_name=='MMSE': # Check number of files
                    if len([f for f in os.listdir(join(Folder_Drive_name,subject_to_uncompress))if os.path.isfile(join(join(Folder_Drive_name,subject_to_uncompress), f))])<2:
                        print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(DataPath)}')           
                        archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
                    else:# The subject folder is complete, therefore go to the next one
                            continue 

            else:
                print(f'=>[DataManager] Uncompressing {subject_to_uncompress}... {idx+1}/{len(DataPath)}')           
                archive.extractall(path=Folder_Drive_name,members=[i for i in archive.namelist() if subject_to_uncompress in i])   
        print(f'=>[DataManager] {len(DataPath)} subjects loaded succesfully in {Folder_Drive_name}')   

    return DataPath

def save_subjects_metadata(args,df:pd.DataFrame,rootS:str,rootL:str=r'/content/data', name:str='Dataset_subjects.txt', VERBOSE:int=1, is_COLAB:str=False):
    """
    Function to save all the subjects_names and the number of frames
    Args:
        rootS (str) : Path where the file will be saved
        rootL (str) : Where the real data is located (Default the Google Driveone)
        name (str): Name of the file to be saved
        VERBOSE (int): the higher the more info about the process will be printed
        is_COLAB (bool): Flag indicating that we are working in Google Colab
    """ 
    if is_COLAB==True:
        rootL = r'/content/data'
    
    if VERBOSE>0: print(f'=>[save_subjects_metadata] Saving {name} in {rootS}...')
    fout = os.path.join(os.path.abspath(rootS),name)
    fo = open(fout, "w")
    for idx,i in df.iterrows():
        if args.network in ['LSTMDFMTM128']:
            file_name = os.path.join(os.path.abspath(rootL),i['subject'])
            file = np.array(scipy.io.loadmat(file_name)['Pred'][0])
        else:
            file_name = os.path.join(os.path.abspath(rootL),i['subject'],i['subject']+'.npy')
            file = np.load(file_name)
        fo.write('{}: {} frames\n'.format(i['subject'],len(file)))
    fo.close()
    if VERBOSE>0: print('=>[save_subjects_metadata] Done.')


def getSubjectIndependentTrainValDataFrames(args, df:pd.DataFrame, VERBOSE:int):
    '''
    Function to set Train and Test sets Metadata from a global DataFrame with the full dataset metadata
    Args:
        args (argparse): Parameters to set the data
        df (pd.DataFrame): With the subjects names of the data
        VERBOSE (int): Flag to show internal procedure of this function
    Return:
        train_data, test_data (DataFrame): Subjects to be used for trining and testing
    '''

    if args.is_reproducible:     
        rng = np.random.RandomState(args.seed)
    else:
        rng = np.random.RandomState(hash(datetime.now())% 2**32 - 1) 

    if args.is_from_rPPG: # Managing rPPG signals as input
        """
        Here, we are going to train from predictions given by another model, that means that the input expected is 
        5 folders called 0,1,2,3, and 4 with the rPPG signals.
        """
        TR_subjects_path_list = []; VAL_subjects_path_list = []
        TR_subjects_names_list = []; VAL_subjects_names_list = []

        for fold in ['0','1','2','3','4']:
            crt_fold_path = join(abspath(args.load_dataset_path),fold)
            if fold == str(args.fold): # If is the fold selected it means that this is my validation data
                for subject in os.listdir(crt_fold_path):
                    if os.path.isfile(join(crt_fold_path,subject)):
                        if args.database_name=='VIPL' and subject.startswith('p')  and subject.endswith('.mat'):
                            VAL_subjects_names_list.append(subject)
                            VAL_subjects_path_list.append(crt_fold_path)
                            
            else: # the rest is my training data
                for subject in os.listdir(crt_fold_path):
                    if os.path.isfile(join(crt_fold_path,subject)):
                        if args.database_name=='VIPL' and subject.startswith('p')  and subject.endswith('.mat'):
                            TR_subjects_names_list.append(subject)
                            TR_subjects_path_list.append(crt_fold_path)
                
        train_data = pd.DataFrame({'path':TR_subjects_path_list,'subject':TR_subjects_names_list}) 
        test_data = pd.DataFrame({'path':VAL_subjects_path_list,'subject':VAL_subjects_names_list}) 
        
        # Uncomment to save the train and test subjects
        #np.savetxt(join(abspath(args.save_path),'train_f'+str(args.fold)+'.txt'), train_data.subject, fmt='%s')
        #np.savetxt(join(abspath(args.save_path),'test_f'+str(args.fold)+'.txt'), test_data.subject, fmt='%s')

        # DEBUG OPTIONS
        if args.dataset_percentage < 0:#Here I want to overfit the network with a few samples
            train_data = train_data[0:int(abs(args.dataset_percentage))]
            test_data = train_data.copy()
        
    else: # Managing video-frames as input
        
        # Shuffle subjects before the spliting (in order to not take closer subjects to the initial distribution)
        df =  shuffle(df,random_state=rng)
        
        # CREATE FOLDER OF CURRENT FOLD
        if (not(args.is_SWEEP) and not(exists(join(abspath(args.save_path),str(args.fold))))): os.makedirs(join(abspath(args.save_path),str(args.fold)))
            
        if args.dataset_percentage < 0:
            """Taking one single subject: Only for debug"""
            train_data = df
            test_data = df
        
        else: # Using more than one subject
    
            if args.is_5050_validation:#If we are in 50_50 validation instead of cross validation
                """
                50_50 EXPERIMENT, 50% training and 50% testing
                """ 
                assert (args.fold == 0) or (args.fold == 1), f'=>[getTrainValDataFrames] ERROR! Fold {args.fold} not Available in 5050 validation, please use 0 or 1'
    
                if VERBOSE>0: print(f'=>[getTrainValDataFrames] 50_50 validation Fold {args.fold}')
                if args.fold==0:
                    # STRATIFIED TRAIN_TEST_SPLIT
                    try:
                        train_data, test_data = train_test_split(df,train_size=0.5,random_state=rng, stratify=df['scenario'])
                    except:
                        if VERBOSE>0: print('=>[getTrainValDataFrames] Impossible to split stratified, using simple split')             
                        train_data, test_data = train_test_split(df,train_size=0.5,random_state=rng)       
                elif args.fold==1:
                    # STRATIFIED TRAIN_TEST_SPLIT (I INVERSE TRAIN_DATA AND TEST_DATA COMPARED WITH FOLD 0)
                    try:
                        test_data, train_data = train_test_split(df,train_size=0.5,random_state=rng, stratify=df['scenario'])
                    except:
                        if VERBOSE>0: print('=>[getTrainValDataFrames] Impossible to split stratified, using simple split')             
                        test_data, train_data = train_test_split(df,train_size=0.5,random_state=rng)   
            
            else: # If we are in train_test_split, cross validation or training in the full dataset
    
                assert args.fold in [0,1,2,3,4,-1,-2], f'=>[getTrainValDataFrames] ERROR! Fold {args.fold} not available. Please use 0,1,2,3,4,-1 or -2'            
                
                if args.fold == -2: 
                    """
                    TRAINING IN FULL DATASET 100%
                    """   
                    if VERBOSE>0: print('=>[getTrainValDataFrames] Training in 100% of data') 
                    train_data = df
                    test_data = df
                
                elif args.fold == -1:
                    """
                    TRAIN_TEST_SPLIT 80/20
                    """        
                    if VERBOSE>0: print('=>[getTrainValDataFrames] Train_test_split: 80% training, 20% testing') 
        
                    # STRATIFIED TRAIN_TEST_SPLIT
                    try:
                        train_data, test_data = train_test_split(df,train_size=0.8,random_state=rng, stratify=df['scenario'])
                    except:
                        if VERBOSE>0: print('=>[getTrainValDataFrames] Impossible to split stratified, using simple split') 
                        train_data, test_data = train_test_split(df,train_size=0.8,random_state=rng)            
      
                elif args.fold in [0,1,2,3,4]:
                    """
                    TRAINING IN 4 FOLDS AND TESTING IN THE REMAINING ONE
                    """  
                    if VERBOSE>0: print(f'=>[getTrainValDataFrames] Fold {args.fold} of 5-fold cross-validation')
                    contFold = 0
                    #Try stratified
                    try:
                        kf = StratifiedKFold(n_splits=5,random_state=rng,shuffle=True)
                        kf.get_n_splits(df,df['scenario'])
                        # TAKE DATA DEPENDING ON FOLD         
                        for train_index, val_index in kf.split(df,df['scenario']):
                            if args.fold == contFold:
                                train_data = df.iloc[train_index]#train_ds = train_data.loc[train_index,:].copy()
                                test_data = df.iloc[val_index]#test_ds = train_data.loc[val_index,:].copy()
                            contFold = contFold + 1                    
                    except:
                        if VERBOSE>0: print('=>[getTrainValDataFrames] Impossible to split stratified, using simple split') 
                        kf = KFold(n_splits=5) 
                        kf.get_n_splits(df)
                        # TAKE DATA DEPENDING ON FOLD         
                        for train_index, val_index in kf.split(df):
                            if args.fold == contFold:
                                train_data = df.iloc[train_index]#train_ds = train_data.loc[train_index,:].copy()
                                test_data = df.iloc[val_index]#test_ds = train_data.loc[val_index,:].copy()
                            contFold = contFold + 1                      
    
    
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)
        
        if args.save_subjects_metadata>0:
            save_subjects_metadata(train_data, args.save_path, abspath(args.load_dataset_path), f'train_subjects_f{args.fold}.txt', VERBOSE, args.in_COLAB)
            save_subjects_metadata(test_data, args.save_path, abspath(args.load_dataset_path), f'test_subjects_f{args.fold}.txt', VERBOSE, args.in_COLAB)
    
    return train_data, test_data

def save_windows_metadata(is_reproducible, in_COLAB, load_dataset_path, save_path, config, train_df, window, step_tr, db_name, seed_dataset, VERBOSE):
    """
    Helper Function to save the the windows metadata using during training in a .txt file.
    This is done to debug the reproducibility of the code
    """
    if VERBOSE>0: print(f'=>[save_windows_metadata] This may spend a few minutes. You can avoid this process by not using  save_windows_metadata flag')
    trainds = PhysNetTrainDataset(is_reproducible, in_COLAB, load_dataset_path,  train_df, window, step_tr,True, db_name, seed_dataset)
    trainds.ShuffleWindowsMetadata()
    traindsloader = DataLoader(trainds, config.batch_size, drop_last=False, shuffle=False)
    
    nameTr_windows = 'TrainFirstWindowMetaData.txt' 
    
    fout = join(abspath(save_path),nameTr_windows)
    fo = open(fout, "w")        
    for sample in traindsloader:
        indexes = sample['index']
        subject_names = sample['name']            
        for i in range(0,len(subject_names)):
            subject_name = subject_names[i]
            fo.write(f'{subject_name}: [{indexes[0][i]},{indexes[1][i]})\n')
    fo.close()    
    if VERBOSE>0: print(f'=>[save_windows_metadata] Done')
    
if __name__ == '__main__':
    print(1)
    pass