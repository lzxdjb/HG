import torch
i = 10
gradient = torch.rand((1  , 3), device='cuda' , dtype=torch.float64)
print(gradient.shape)
