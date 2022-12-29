import torch.nn as nn
import torch

def hrmae_hrr_snr():
    """
    This metric will measure  
    """
    def __init__(self):
        super(hrmae_hrr_snr,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_to_be_minimized = True

    def forward(self, sample):
        assert len(sample)==3, print('=>[NP_NSNR] ERROR, sample must have 3 values [y_hat, y, time]')
        #TODO    
        
    
    def get_HRyhat_HRy_yhatsnr_slidingwindow(self):
        """This function use a sliding window to get HR of yhat, HR of y and SNR from yhat"""
        #TODO
        pass

def stand_alone(Run=False):
    """ Function to test the metric
    """
    if Run:
        print('stand_alone not yet implemented')
        #TODO


if __name__ == '__main__':
    stand_alone()
