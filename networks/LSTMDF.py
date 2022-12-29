import torch.nn as nn
import torch


#%% LSTMDF-MTM  PYTORCH 
# 2022/03/16 We want to propose a LSTMDF-like but with 128 as Temporal input
class LSTMDFMTM125(nn.Module):
    def __init__(self, input_size=1, hidden_size=125, output_size=125, device='auto'):
        '''
        1)input_size: Corresponds to the number of features in the input (1)
        2)hidden_size: Specifies the number of hidden neurons
        3)output_size: The number of items in the output, since we want to predict the full sequence
            the output size will be 125.

        '''
        super().__init__()
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm_1 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size) # Fully connected
        self.reset_hidden()

    def reset_hidden(self):
        self.hn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        self.cn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]        

    def init_weights(self):
        print('=>[LSTMDFMTM128] Initializating weights')
        for (name_layer,layer) in self.named_children():
            # LSTM WEIGHTS AND BIAS INITIALIZATION
            if name_layer.startswith('lstm'):
                """
                Use orthogonal init for recurrent layers, xavier uniform for input layers
                Bias is 0 (except for forget gate for first LSTM layer)
                """
                for name_param, param in layer.named_parameters():
                    if 'weight_ih' in name_param:# input
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif 'weight_hh' in name_param:# recurrent
                        torch.nn.init.orthogonal_(param.data,gain=1)
                    elif 'bias' in name_param:# Bias
                        torch.nn.init.zeros_(param.data)
                        if name_layer=='lstm':
                            param.data[layer.hidden_size:2 * layer.hidden_size] = 1#unit_forget_bias=True
                        
            elif name_layer == 'linear':
                for name_param, param in layer.named_parameters():
                    if name_param.startswith('weight'):
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif name_param.startswith('bias'):
                        torch.nn.init.zeros_(param.data)

    def forward(self, x):# [b,T=125,F=1]
        # lstm is Stateful and Return_sequences=True
        x_longer = x.view(1,x.shape[0]*x.shape[1],x.shape[2])#[b, b*125,1] -- [1,125*128,1] 
        out_longer, (self.hn, self.cn) = self.lstm(x_longer, (self.hn.detach(), self.cn.detach()))#[1,b*125,125] --[1,125*128,125] 
        out = out_longer.view(x.shape[0],x.shape[1],out_longer.shape[2])#[b, 125, 125] -- [128,125,125]
        out = self.dropout(out[:,:,:]) 
        #lstm_1 is Stateless and Return_sequences=True
        hn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_1(out,(hn1,cn1))#[b, 125, 125] -- [128,125,125]
        out = self.dropout_1(out[:,:,:])
        #lstm_2 is Stateless and Return_sequences=False
        hn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_2(out,(hn2,cn2))#[b,125,125]  --  [128,125,125]
        out = out[:,-1,:]#Return_sequences=False. [b,125] [128,125]
        # Dense with linear activation
        out = self.linear(out)#[b,125] -- [128,125]
        return out.unsqueeze(-1), _, _, _

# %% LSTMDFMTM128 - Same as LSTMDFMTM125 but with input of 128 frames
class LSTMDFMTM128(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=128, device='auto'):
        '''
        1)input_size: Corresponds to the number of features in the input (1)
        2)hidden_size: Specifies the number of hidden neurons
        3)output_size: The number of items in the output, since we want to predict the full sequence
            the output size will be 125.

        '''
        super().__init__()
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm_1 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size) # Fully connected
        self.reset_hidden()

    def reset_hidden(self):
        self.hn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        self.cn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]        

    def init_weights(self):
        print('=>[LSTMDFMTM128] Initializating weights')
        for (name_layer,layer) in self.named_children():
            # LSTM WEIGHTS AND BIAS INITIALIZATION
            if name_layer.startswith('lstm'):
                """
                Use orthogonal init for recurrent layers, xavier uniform for input layers
                Bias is 0 (except for forget gate for first LSTM layer)
                """
                for name_param, param in layer.named_parameters():
                    if 'weight_ih' in name_param:# input
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif 'weight_hh' in name_param:# recurrent
                        torch.nn.init.orthogonal_(param.data,gain=1)
                    elif 'bias' in name_param:# Bias
                        torch.nn.init.zeros_(param.data)
                        if name_layer=='lstm':
                            param.data[layer.hidden_size:2 * layer.hidden_size] = 1#unit_forget_bias=True
                        
            elif name_layer == 'linear':
                for name_param, param in layer.named_parameters():
                    if name_param.startswith('weight'):
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif name_param.startswith('bias'):
                        torch.nn.init.zeros_(param.data)

    def forward(self, x):# [b,T=128,F=1]
        # lstm is Stateful and Return_sequences=True
        x_longer = x.view(1,x.shape[0]*x.shape[1],x.shape[2])#[b, b*125,1] -- [1,125*128,1] 
        out_longer, (self.hn, self.cn) = self.lstm(x_longer, (self.hn.detach(), self.cn.detach()))#[1,b*125,125] --[1,125*128,125] 
        out = out_longer.view(x.shape[0],x.shape[1],out_longer.shape[2])#[b, 125, 125] -- [128,125,125]
        out = self.dropout(out[:,:,:]) 
        #lstm_1 is Stateless and Return_sequences=True
        hn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_1(out,(hn1,cn1))#[b, 125, 125] -- [128,125,125]
        out = self.dropout_1(out[:,:,:])
        #lstm_2 is Stateless and Return_sequences=False
        hn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_2(out,(hn2,cn2))#[b,125,125]  --  [128,125,125]
        out = out[:,-1,:]#Return_sequences=False. [b,125] [128,125]
        # Dense with linear activation
        out = self.linear(out)#[b,125] -- [128,125]
        return out.unsqueeze(-1), _, _, _
    
# %% LSTMDFMTO128 - Same as LSTMDFMTM128 but with output of 1 value

class LSTMDFMTO128(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, device='auto'):
        '''
        1)input_size: Corresponds to the number of features in the input (1)
        2)hidden_size: Specifies the number of hidden neurons
        3)output_size: The number of items in the output, since we want to predict the full sequence
            the output size will be 125.

        '''
        super().__init__()
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm_1 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.output_size) # Fully connected
        self.reset_hidden()

    def reset_hidden(self):
        self.hn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        self.cn = torch.zeros(1, 1, self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]        

    def init_weights(self):
        print('=>[LSTMDFMTM128] Initializating weights')
        for (name_layer,layer) in self.named_children():
            # LSTM WEIGHTS AND BIAS INITIALIZATION
            if name_layer.startswith('lstm'):
                """
                Use orthogonal init for recurrent layers, xavier uniform for input layers
                Bias is 0 (except for forget gate for first LSTM layer)
                """
                for name_param, param in layer.named_parameters():
                    if 'weight_ih' in name_param:# input
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif 'weight_hh' in name_param:# recurrent
                        torch.nn.init.orthogonal_(param.data,gain=1)
                    elif 'bias' in name_param:# Bias
                        torch.nn.init.zeros_(param.data)
                        if name_layer=='lstm':
                            param.data[layer.hidden_size:2 * layer.hidden_size] = 1#unit_forget_bias=True
                        
            elif name_layer == 'linear':
                for name_param, param in layer.named_parameters():
                    if name_param.startswith('weight'):
                        torch.nn.init.xavier_uniform_(param.data,gain=1)
                    elif name_param.startswith('bias'):
                        torch.nn.init.zeros_(param.data)

    def forward(self, x):# [b,T=125,F=1]
        # lstm is Stateful and Return_sequences=True
        x_longer = x.view(1,x.shape[0]*x.shape[1],x.shape[2])#[b, b*125,1] -- [1,125*128,1] 
        out_longer, (self.hn, self.cn) = self.lstm(x_longer, (self.hn.detach(), self.cn.detach()))#[1,b*125,125] --[1,125*128,125] 
        out = out_longer.view(x.shape[0],x.shape[1],out_longer.shape[2])#[b, 125, 125] -- [128,125,125]
        out = self.dropout(out[:,:,:]) 
        #lstm_1 is Stateless and Return_sequences=True
        hn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn1 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_1(out,(hn1,cn1))#[b, 125, 125] -- [128,125,125]
        out = self.dropout_1(out[:,:,:])
        #lstm_2 is Stateless and Return_sequences=False
        hn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        cn2 = torch.zeros(1, x.shape[0], self.hidden_size).to(self.device)#[num_layers*num_directions,batch,hidden_size]
        out, (_,_) = self.lstm_2(out,(hn2,cn2))#[b,125,125]  --  [128,125,125]
        out = out[:,-1,:]#Return_sequences=False. [b,125] [128,125]
        # Dense with linear activation
        out = self.linear(out)#[b,1]
        return out.unsqueeze(-1), _, _, _    