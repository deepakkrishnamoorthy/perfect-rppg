try:
    if 'google.colab' in str(get_ipython()):
        COLAB = True
    else:
        COLAB = False
except:
    COLAB = False

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy import signal

if not(COLAB):
    from utils.detect_peaks import detect_peaks
else:
    from deep_rppg.utils.detect_peaks import detect_peaks


if __name__ == "__main__":
    print('hola')
    # get_HR_SNR_r_MSE(1,2,3,4,5,6,7,8)
    

def getHRandSNR(pulseTrace,gtTrace,Fs=25,winLengthSec=15,stepSec=0.5,lowF=0.7,upF=3.5,VERBOSE=0):
    '''
    Parameters
    ----------
    pulseTrace : array
        rppg traces.
    gtTrace : array
        ground truth traces.
    Fs : int, optional
        Frequency of rppg and gt. The default is 25.
    winLengthSec : int, optional
        Length of the sliding windows in seconds. The default is 15.        
    stepSec : int, optional
        length of the step to take between windows, in seconds. The default is 0.5. 
    lowF : int, optional
        low frequency for HR measurement. The default is 0.7
    upF : int, optinal
        up frequency for HR measurement. The defaul is 3.5
    Returns
    -------
    rPPG_HR: Heart rate of rPPG traces per window.
    PPG_HR: Heart rate of PPG traces per window.
    rPPG_SNR: Signal to noise Ratio of rPPG.
    PPG_SNR: Signal to noise Ratio of PPG
    r: Pearson's correlation between rPPG and PPG

    '''
    #IF rppg is exactly winLengthSec, add one more value to get into the function
    if np.size(pulseTrace)<=winLengthSec*Fs:
        if np.size(pulseTrace)<winLengthSec*Fs:
            print('=>[getHRandSNR]Can not measure metrics because signals is shorter than %i seconds'%winLengthSec)
            return [],[],[]
        elif np.size(pulseTrace)==winLengthSec*Fs:
            pulseTrace = np.append(pulseTrace,pulseTrace[-1]).copy()
            gtTrace = np.append(gtTrace,gtTrace[-1]).copy()
            
    
    pulseTrace = np.asarray(pulseTrace).copy()
    gtTrace = np.asarray(gtTrace).copy()
    # CALCULE Timetrace of rPPG with its frequency 
    timeTrace = np.zeros(pulseTrace.size)
    for j in range(0,len(timeTrace)):
        timeTrace[j] = j*(1/Fs)
    
    # Calculate timeTrace of PPG with its frequency
    gtTime = timeTrace
    
    traceSize = len(pulseTrace);
    winLength = round(winLengthSec*Fs)# length of the sliding window for FFT
    step = round(stepSec*Fs);# length of the steps for FFT
    halfWin = (winLength/2);
    
    show1window = True
    rPPG_SNR = []; PPG_SNR = []
    rPPG_HR = []; PPG_HR = []
    Pearsonsr = []
    MSE = []
    cont=0
    for i in range(int(halfWin),int(traceSize-halfWin),int(step)):#for i=halfWin:step:traceSize-halfWin
        #Uncomment next three lines just to debug
        #if cont == 90:
        #    print('error')
        #print(cont);cont=cont+1
        
        ###
        # GET CURRENT WINDOW
        ## get start/end index of current window
        startInd = int(i-halfWin) #startInd = i-halfWin+1;
        endInd = int(i+halfWin) # endInd = i+halfWin;
        startTime = int(timeTrace[startInd]) # startTime = timeTrace(startInd);
        endTime = int(timeTrace[endInd]) #timeTrace(endInd);
        # get current pulse window
        crtPulseWin = pulseTrace[startInd:endInd]# crtPulseWin = pulseTrace(startInd:endInd);
        crtTimeWin = timeTrace[startInd:endInd]# crtTimeWin = timeTrace(startInd:endInd);
        # get current PPG window
        startIndGt = startInd # [~, startIndGt] = min(abs(gtTime-startTime));
        endIndGt = endInd # [~, endIndGt] = min(abs(gtTime-endTime));
        crtPPGWin = gtTrace[startIndGt:endIndGt]
        crtTimePPGWin = gtTime[startIndGt:endIndGt]
        # get exact PPG Fs
        # Fs_PPG = 1/mean(diff(crtTimePPGWin));       
        if VERBOSE>0 and show1window==True: pltnow(crtPulseWin,crtPPGWin,val=2,fr=Fs)
        
        #########################
        # rPPG: SPECTRAL ANALYSIS
        ### rPPG: Get spectrum by Welch
        # Get power spectral density in Frequency of HR in humans [0.7-3.5]
        rppg_freq_w, rppg_power_w = signal.welch(crtPulseWin, fs=Fs)
        rppg_freq2 = [item1 for item1 in rppg_freq_w if item1 > lowF and item1 < upF]
        rppg_power2 = [item2 for item1,item2 in zip(rppg_freq_w,rppg_power_w) if item1 > lowF and item1 < upF]
        rppg_freq_w = rppg_freq2.copy();rppg_power_w = rppg_power2.copy()
        # Find highest peak in the spectral density and take its frequency value
        loc = detect_peaks(np.asarray(rppg_power_w), mpd=1, edge='rising',show=False)
        if loc.size == 0 :# If no peak was found
            loc = np.array([0])
        loc = loc[np.argmax(np.array(rppg_power_w)[loc])]#just highest peak

        rPPG_peaksLoc = np.asarray(rppg_freq_w)[loc]
        if VERBOSE>0 and show1window==True: 
            plt.figure(),plt.title('rPPG Spectrum,welch'),plt.plot(rppg_freq_w,rppg_power_w)
            plt.axvline(x=rPPG_peaksLoc,ymin=0,ymax=1,c='r'),plt.show(),plt.pause(1)
                
        # YB: SNR is more intersting with FFT so we get spectra again (I don't care about processing cost :)        
        width = 0.4
        # rPPG: Get spectrum by FFT        
        N = len(crtPulseWin)*3
        rppg_freq = np.arange(0,N,1)*Fs/N#freq=[0 : N-1]*Fs/N;
        rppg_power = np.abs(np.fft.fft(crtPulseWin,N))**2#power = abs(fft(x,N)).^2;
        rppg_freq2 = [item1 for item1 in rppg_freq if item1 > lowF and item1 < upF]
        rppg_power2 = [item2 for item1,item2 in zip(rppg_freq,rppg_power) if item1 > lowF and item1 < upF]
        rppg_freq = rppg_freq2.copy();rppg_power = rppg_power2.copy()
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG Spectrum,FFT'),plt.plot(rppg_freq,rppg_power),plt.show(),plt.pause(1)        
        
        #rPPG: SNR
        range1 = [((i>(rPPG_peaksLoc-width/2))and(i<(rPPG_peaksLoc+width/2))) for i in rppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range 1'),plt.plot(rppg_freq,range1),plt.show(),plt.pause(1)
        range2 = [((i>((rPPG_peaksLoc*2)-(width/2)))and(i<((rPPG_peaksLoc*2)+(width/2)))) for i in rppg_freq]
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range 2'),plt.plot(rppg_freq,range2),plt.show(),plt.pause(1)
        rango = np.logical_or(range1, range2)
        #if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:range'),plt.plot(rppg_freq,rango),plt.show(),plt.pause(1)
        Signal = rppg_power*rango # signal = rPPG_power.*range;
        Noise = rppg_power*~rango #noise = rPPG_power.*(~range);
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:Signal'),plt.plot(rppg_freq,Signal),plt.show(),plt.pause(1)
        if VERBOSE>0 and show1window==True: plt.figure(),plt.title('rPPG:Noise'),plt.plot(rppg_freq,Noise),plt.show(),plt.pause(1)
        n = np.sum(Noise) # n = sum(noise);
        s = np.sum(Signal) # s = sum(signal);
        snr = 10*np.log10(s/n) # snr(ind) = 10*log10(s/n);
        rPPG_SNR.append(snr)
        
        #rPPG: HR
        rPPG_HR.append(rPPG_peaksLoc*60)#rPPG_peaksLoc(1)*60;
        
        #########################
        # PPG: SPECTRAL ANALYSIS
        ### PPG: Get spectrum by Welch       
        # Get power spectral density in Frequency of HR in humans [0.7-3.5]
        ppg_freq_w, ppg_power_w = signal.welch(crtPPGWin, fs=Fs)
        ppg_freq2 = [item1 for item1 in ppg_freq_w if item1 > lowF and item1 < upF]
        ppg_power2 = [item2 for item1,item2 in zip(ppg_freq_w,ppg_power_w) if item1 > lowF and item1 < upF]
        ppg_freq_w = ppg_freq2.copy();ppg_power_w = ppg_power2.copy()
        # Find highest peak in the spectral density and take its frequency value
        loc = detect_peaks(np.asarray(ppg_power_w), mpd=1, edge='rising',show=False)
        if loc.size == 0 :# If no peak was found
            loc = np.array([0])
        loc = loc[np.argmax(np.array(ppg_power_w)[loc])]#just highest peak
        PPG_peaksLoc = np.asarray(ppg_freq_w)[loc]
        if VERBOSE>0 and show1window==True:
            plt.figure(),plt.title('PPG Spectrum,welch'),plt.plot(ppg_freq_w,ppg_power_w)
            plt.axvline(x=PPG_peaksLoc,ymin=0,ymax=1,c='r'),plt.show(),plt.pause(1)
        
        #PPG: HR
        PPG_HR.append(PPG_peaksLoc*60)#rPPG_peaksLoc(1)*60;
        
        show1window=False # Just plot first window

    return np.asarray(rPPG_HR),np.asarray(PPG_HR),np.asarray(rPPG_SNR)

def pltnow(variable,variable2=0,val=0,fr=25):
    sr = 1/fr #sf = sampling frequency
    #sr = sampling rate
    if val==0: #simpleplot
        plt.figure(),plt.plot(variable),plt.show(),plt.pause(0.05)
        plt.ylabel('Amplitude')
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
    elif val==1: #generate traces
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        plt.figure(),plt.plot(timeTrace,variable),plt.show(),plt.pause(0.05)
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
    elif val==2:
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        plt.figure()
        line1, = plt.plot(timeTrace,variable)
        line2, = plt.plot(timeTrace,variable2)
        plt.xlim(timeTrace[0], timeTrace[-1])
        plt.legend([line1, line2], ['signal 1', 'signal 2'], loc='upper right')
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')        
        plt.show(),plt.pause(0.05)
    elif val==3:#plot different-length-signals
        timeTrace = np.zeros(len(variable))
        for j in range(0,len(timeTrace)):
            timeTrace[j] = j*sr#sr=0.04
        timeTrace2 = np.zeros(len(variable2))
        for j in range(0,len(timeTrace2)):
            timeTrace2[j] = j*sr#sr=0.04
        plt.figure()
        plt.plot(timeTrace,variable)
        plt.plot(timeTrace2,variable2)
        plt.xlim(timeTrace[0], np.max((timeTrace[-1],timeTrace2[-1])))
        #plt.legend([line1, line2], ['signal 1', 'signal 2'], loc='upper right')
        plt.xlabel('time [s]'),plt.ylabel('Amplitude')        
        plt.show(),plt.pause(0.05)
        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()
        