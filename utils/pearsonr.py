import numpy as np
import torch
def r(x,y):
    x
    """PEARSONS CORRELATION"""
    sum_x = torch.sum(torch.tensor(x))
    sum_y = torch.sum(torch.tensor(y))
    sum_xy = torch.sum(torch.tensor(x)*torch.tensor(y))
    sum_x2 = torch.sum(torch.pow(torch.tensor(x),2))
    sum_y2 = torch.sum(torch.pow(torch.tensor(y),2))
    N = torch.tensor(x).shape[0]
    pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
    return pearson.item()
