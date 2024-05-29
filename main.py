import torch
from graph import OptimizeProblem
import numpy as np


# Define system matrices A and B
state_size = 3
control_size = 2

A = np.identity(state_size)
B = np.zeros((state_size, control_size))

Q = np.identity((state_size , state_size)) # 
R = np.identity((control_size , control_size)) # 

# Convert the NumPy arrays to PyTorch tensors
A_tensor = torch.tensor(A, dtype=torch.float32)
B_tensor = torch.tensor(B, dtype=torch.float32)
Q_tensor = torch.tensor(Q, dtype=torch.float32)
R_tensor = torch.tensor(R, dtype=torch.float32)

# Move the tensors to the GPU
A_cuda = A_tensor.cuda()
B_cuda = B_tensor.cuda()
Q_cuda = Q_tensor.cuda()
R_cuda = R_tensor.cuda()

T = 0.2
horizon = 10

x_initial = torch.tensor([0.0, 0.0 , np.sin(0)], device='cuda')
x_final = torch.tensor([0.0, 0.0 , np.sin(0)], device='cuda')


if __name__ == "__main__":
    # Define time grid
    t_grid = torch.linspace(0, 1, steps=11).cuda()
    print("t_grid = " ,t_grid.shape)

    # Initial and final states
    x_initial = torch.tensor([1.0, 0.0], device='cuda' , requires_grad= True)
    x_final = torch.tensor([0.0, 0.0], device='cuda' , requires_grad= True)

    # Control bounds (u_min, u_max)
    control_bounds = torch.tensor([-1.0], device='cuda'), torch.tensor([1.0], device='cuda' ,  requires_grad= True)

    # State bounds (x_min, x_max)
    state_bounds = torch.tensor([-1.0, -1.0], device='cuda'), torch.tensor([1.0, 1.0], device='cuda' ,  requires_grad= True)

    Q = torch.eye(state_bounds[0].size(0)).cuda()  
    # print("Q = " , Q)

    R = torch.eye(control_bounds[0].size(0)).cuda()
    # print("R = " , R)

    # Create optimization problem
    # op = OptimizeProblem(id=0, data=None, start_vertex=None, end_vertex=None, A=A, B=B  , Q = Q , R = R)  
    # op.setup_problem(t_grid, x_initial, x_final, control_bounds, state_bounds)

    op = OptimizeProblem(id=0, data=None, start_vertex=None, end_vertex=None, A=A, B=B  , Q = Q , R = Rn, T = T , horizon = horizon)  
    op.setup_problem(t_grid, x_initial, x_final, control_bounds, state_bounds)

    # Print object using __str__ method
    # print(op)

    # Inspect object using __repr__ method
    # print(repr(op))

    # Compute cost
    total_cost = op.compute_cost()
    # op.compute_gradient()
    # print("Total Cost:", total_cost)
