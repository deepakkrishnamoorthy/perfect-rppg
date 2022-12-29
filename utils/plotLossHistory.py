from matplotlib.pyplot import figure,plot,show,pause,xlabel,ylabel,title,legend
import numpy as np

def plotLossHistory(n_epochs:int,train_loss:list, val_loss:list=None):
    """Function to plot Loss history during training
    """
    crt_epoch = len(train_loss)
    xaxis = np.arange(0,n_epochs)
    # train_loss = [1.,2.,3.]; val_loss = [1.5,2.5,3.5]
    train_loss = np.concatenate((np.array(train_loss),np.zeros(n_epochs-len(train_loss))))
    if val_loss != None:
        validation_loss = np.concatenate((np.array(val_loss),np.zeros(n_epochs-len(val_loss))))
    figure(),title('Current epoch: '+str(crt_epoch))    
    plot(xaxis,train_loss)
    if val_loss != None: plot(xaxis,validation_loss)
    xlabel('epochs');ylabel('loss')
    legend(['train','val'])
    show()

