import torch


grad = torch.zeros((1 ,3) , device='cuda' , dtype = torch.float64)

print(grad.size(1))