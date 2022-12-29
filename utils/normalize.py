import numpy as np
#%%
#Fonction normalisation
def normalize(data):
    
    from sklearn.preprocessing import MinMaxScaler
    x = np.asarray(data)
    x = x.reshape(len(x), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(x)
    scaled_x = scaler.transform(x)
    return scaled_x

