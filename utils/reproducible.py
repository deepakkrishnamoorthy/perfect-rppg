import torch
import random
import numpy as np

#%%
def reproducible(seed):## Uncomment to reproducible results by deterministic behaviour?
    '''Function used to manage same seed in Pytorch'''
    print(f'=>[REPLICABLE] True, with seed {seed}')
    print('    =>WARNING: SLOWER THIS WAY')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False      
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # rng = np.random.RandomState(seed)
    # return rng
