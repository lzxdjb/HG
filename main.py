import torch
from src.graph import NaiveQuestion
from src.draw import Draw_MPC_point_stabilization_v1
import numpy as np


# Define system matrices A and B
state_size = 3
control_size = 2

A = np.identity(state_size)
B = np.zeros((state_size, control_size))

Q = np.identity(state_size) #
Q  = Q * 1
# Q[2 , 2] = 100

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

T = 0.1
horizon = 10

x_initial = torch.tensor([[0.0, 0.0 , 0]], device='cuda' , dtype=torch.float64)
x_final = torch.tensor([[2, 2 , np.pi / 3]], device='cuda' , dtype=torch.float64)

upper_state_constrain = torch.tensor([2 , 2 , np.sin(np.pi / 2)])
lower_state_constrain = torch.tensor([-2 , -2 , 0])


if __name__ == "__main__":
    # Define time grid


    op = NaiveQuestion(id=0, data=None, start_vertex=None, end_vertex=None, A=A, B=B  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size)  
    
    op.setup_problem(x_initial, x_final, upper_state_constrain,lower_state_constrain)

    # Print object using __str__ method
    # print(op)

    # Inspect object using __repr__ method
    # print(repr(op))

    # Compute cost
    # total_cost = op.ShootingCost()
    # op.SolveMultiShooting(0.001)
    xx = op.Leverberg()

    x_initial = x_initial.cpu().numpy()
    x_final = x_final.cpu().numpy()

    # Assuming xx is a list of tensors on CUDA
    xx = [tensor.cpu().numpy().reshape(3 , 1) for tensor in xx]

    # print("x_final = " , x_final.shape)
    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=0.3, init_state=x_initial.reshape(3 , 1), target_state=x_final.reshape(3 , 1), robot_states=xx
    )
    # print("Total Cost:", total_cost)

