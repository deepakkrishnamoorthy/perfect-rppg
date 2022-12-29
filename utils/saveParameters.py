import os
#%%
def saveParameters(args,path,name='parameters'):
    
    with open(os.path.join(path,f'args_f{args.fold}.txt'), 'w') as fp:
        for arg in vars(args):
            fp.write(f'{arg} : {getattr(args, arg)}\n')        
  

