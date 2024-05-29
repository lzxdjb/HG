import torch
from src.graph import OptimizeProblem , NaiveQuestion
from src.draw import Draw_MPC_point_stabilization_v1
import numpy as np


# Define system matrices A and B
state_size = 3
control_size = 2

A = np.identity(state_size)
B = np.zeros((state_size, control_size))

Q = np.identity(state_size) # 
R = np.identity(control_size) # 

# Convert the NumPy arrays to PyTorch tensors
A = torch.tensor(A, dtype=torch.float64)
B = torch.tensor(B, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)

# Move the tensors to the GPU
A = A.cuda()
B = B.cuda()
Q = Q.cuda()
R = R.cuda()

T = 0.2
horizon = 10

x_initial = torch.tensor([0.0, 0.0 , np.sin(0)], device='cuda')
x_final = torch.tensor([0.0, 0.0 , np.sin(np.pi)], device='cuda')


if __name__ == "__main__":
    # Define time grid

    # Control bounds (u_min, u_max)
    control_bounds = torch.tensor([-1.0], device='cuda'), torch.tensor([1.0], device='cuda' ,  requires_grad= True)

    # State bounds (x_min, x_max)
    state_bounds = torch.tensor([-1.0, -1.0], device='cuda'), torch.tensor([1.0, 1.0], device='cuda' ,  requires_grad= True)


    # print("R = " , R)

    # Create optimization problem
    # op = OptimizeProblem(id=0, data=None, start_vertex=None, end_vertex=None, A=A, B=B  , Q = Q , R = R)  
    # op.setup_problem(t_grid, x_initial, x_final, control_bounds, state_bounds)

    op = NaiveQuestion(id=0, data=None, start_vertex=None, end_vertex=None, A=A, B=B  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size)  
    op.setup_problem(x_initial, x_final, control_bounds, state_bounds)
    op.SingleShootingCost()

    # Print object using __str__ method
    # print(op)

    # Inspect object using __repr__ method
    # print(repr(op))

    # Compute cost
    # total_cost = op.compute_cost()
    # op.compute_gradient()
    # print("Total Cost:", total_cost)
