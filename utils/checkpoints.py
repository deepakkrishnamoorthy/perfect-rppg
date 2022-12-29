import torch

def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    torch.save(state,filename)
    
# FUNCTION TO LOAD CHEKCPOINT
def load_checkpoint(model,optimizer,PathL_ModelTrained):
    checkpoint = torch.load(PathL_ModelTrained)
    #model.trainable = False
    model.load_state_dict(checkpoint['model_state_dict'])
    # l2_coeff = 0.01
    # for name, param in model.named_parameters():
    #     checkpoint = checkpoint[name]
    #     current_weights = param.data
    #     weight_diff = current_weight - checkpoint
    #     weight_diff += l2_coeff * current_weight
    #     param.data = checkpoint + weight_diff
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

