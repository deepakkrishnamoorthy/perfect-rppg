import numpy as np
from scipy.signal import butter, lfilter

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 

def SNR(data):
    data['snr_rppg'] = ""
    data['snr_gt'] = ""
    for i in range(0,len(data)):        
        rppg = data['rppg'].iloc[i]
        gt = data['gt'].iloc[i]
        snr_gt = signaltonoise(gt, axis = 0, ddof = 0)
        snr_rppg = signaltonoise(rppg, axis = 0, ddof = 0)
        data['snr_rppg'].iloc[i] = snr_rppg
        data['snr_gt'].iloc[i] = snr_gt
    return data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

