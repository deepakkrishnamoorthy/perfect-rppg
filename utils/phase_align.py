from statsmodels.tsa.stattools import ccovf
import numpy as np
from scipy.interpolate import interp1d

#%%

def phase_align(reference, target, roi, res=100):
    '''
    Cross-correlate data within region of interest at a precision of 1./res
    if data is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision 
    Args:
        reference (1d array/list): signal that won't be shifted
        target (1d array/list): signal to be shifted to reference
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    # convert to int to avoid indexing issues
    ROI = slice(int(roi[0]), int(roi[1]), 1)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract mean
    r1 -= r1.mean()
    r2 -= r2.mean()

    # compute cross covariance 
    
    cc = ccovf(r1,r2,demean=False,unbiased=False)
    #cc = ccovf(r1,r2,demean=False) # 2022/04/14 DB: Modified to avoid warning

    # determine if shift if positive/negative 
    if np.argmax(cc) == 0:
        cc = ccovf(r2,r1,demean=False,unbiased=False)
        #cc = ccovf(r2,r1,demean=False) # # 2022/04/14 DB: Modified to avoid warning
        mod = -1
    else:
        mod = 1

    # often found this method to be more accurate then the way below
    return np.argmax(cc)*mod*(1./res)

    # interpolate data onto a higher resolution grid 
    x,r1 = highres(reference[ROI],kind='linear',res=res)
    x,r2 = highres(target[ROI],kind='linear',res=res)

    # subtract off mean 
    r1 -= r1.mean()
    r1 -= r2.mean()

    # compute the phase-only correlation function
    product = np.fft.fft(r1) * np.fft.fft(r2).conj()
    cc = np.fft.fftshift(np.fft.ifft(product))

    # manipulate the output from np.fft
    l = reference[ROI].shape[0]
    shifts = np.linspace(-0.5*l,0.5*l,l*res)

    # plt.plot(shifts,cc,'k-'); plt.show()
    return shifts[np.argmax(cc.real)]

def highres(y,kind='cubic',res=100):
    '''
    Interpolate data onto a higher resolution grid by a factor of *res*
    Args:
        y (1d array/list): signal to be interpolated
        kind (str): order of interpolation (see docs for scipy.interpolate.interp1d)
        res (int): factor to increase resolution of data via linear interpolation
    
    Returns:
        shift (float): offset between target and reference signal 
    '''
    y = np.array(y)
    x = np.arange(0, y.shape[0])
    f = interp1d(x, y,kind='cubic')
    xnew = np.linspace(0, x.shape[0]-1, x.shape[0]*res)
    ynew = f(xnew)
    return xnew,ynew

