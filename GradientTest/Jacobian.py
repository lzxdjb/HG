import torch

# Define the vectors as PyTorch tensors
x1 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
x2 = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
u1 = torch.tensor([3.0, 3.0], requires_grad=True)

# Define the functions
f1 = x2[0] - x1[0] - torch.cos(x1[2]) * u1[0]
f2 = x2[1] - x1[1] - torch.sin(x1[2]) * u1[0]
f3 = x2[2] - x1[2] - u1[1]

# List of functions
functions = [f1, f2, f3]

# Compute gradients and form the Jacobian
jacobian = []
for f in functions:
    # Compute the gradient of f with respect to x1, x2, u1
    grads = torch.autograd.grad(f, [x1, x2, u1], create_graph=True)
    
    # Concatenate gradients to form the Jacobian row
    jacobian.append(torch.cat([g.view(-1) for g in grads]))

# Stack the rows to form the Jacobian matrix
jacobian_matrix = torch.stack(jacobian)

# Print the Jacobian matrix
print(jacobian_matrix)
