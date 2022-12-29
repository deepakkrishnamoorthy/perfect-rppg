"""
2022/04/6
I wanted to know the distribution in time of the VIPL dataset after taking
only the good ground truth files. An histogram of duration of all subjects
"""
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn
import zipfile
import collections
from natsort import natsorted
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

seed = 10
load_path = r'J:\faces\8_8\synchronized\VIPL_npy\MediapipeFromFascascade\YUV'
save_path = r'J:\faces\8_8\synchronized\VIPL_npy\MediapipeFromFascascade\Distribution'
#%%%
# 1) Replicate the process done in Datamanager.py: Create ordered Dataframe with subjects
# Load archive
filename = os.path.join(r'J:\faces\8_8\synchronized\VIPL_npy\MediapipeFromFascascade\YUV\VIPLYUV8_npy.zip')  
archive = zipfile.ZipFile(filename, 'r')
# Take only nonempty folders
subjects_names_list =  [os.path.dirname(x) for x in archive.namelist()]
subjects_names_list = [item for item, count in collections.Counter(subjects_names_list).items() if count > 1]
subjects_names_list = natsorted(subjects_names_list)
scenario_list = [] 
source_list = []
duration_list = [] # Here we will save the duration per subject
duration_list_aprox = []
for name in subjects_names_list:
    scenario_list.append(int(name[name.find('v')+1]))
    source_list.append(int(name[name.find('s')+1]))
    # get duration of subject
    timestamp = np.loadtxt(join(load_path,name,name+'_timestamp.txt'))
    duration_list.append(timestamp[-1])
    duration_list_aprox.append(int(np.floor(timestamp[-1])))

DataPath = pd.DataFrame({'subject':subjects_names_list,'scenario':scenario_list,'source':source_list,'time':duration_list,'time_aprox':duration_list_aprox})
hist = DataPath['time_aprox'].hist(bins=100,legend=True)
#%%%
# TODO
# TODO
# TODO
# Plots for 5-FOLD CROSS VALIDATION
# 2) Replicate the process done in getSubjectIndependentTrainValDataFrames to get the same subjects in CV5F, fold 0,1,2,3,4


# df = DataPath.copy()
# rng = np.random.RandomState(seed)
# df =  shuffle(df,random_state=rng)

# for fold in [0,1,2,3,4]:
#     contFold = 0
#     kf = StratifiedKFold(n_splits=5,random_state=rng,shuffle=True)
#     kf.get_n_splits(df,df['scenario'])
#     # TAKE DATA DEPENDING ON FOLD         
#     for train_index, val_index in kf.split(df,df['scenario']):
#         if fold == contFold:
#             train_data = df.iloc[train_index]#train_ds = train_data.loc[train_index,:].copy()
#             test_data = df.iloc[val_index]#test_ds = train_data.loc[val_index,:].copy()
#         contFold = contFold + 1    

#     train_data = train_data.reset_index(drop=True)
#     test_data = test_data.reset_index(drop=True)
