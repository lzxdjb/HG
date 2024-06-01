import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class MPC(nn.Module):
    def __init__(self ,  A , B,  Q , R , T , horizon , state_shape , control_shape ):
        super(MPC, self).__init__()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.T = T
        self.horizon = horizon

        self.x_initial = torch.tensor([[0.0, 0.0, 0.0]])
        self.x_final = torch.tensor([[2., 2., 2.]])

        self.states = nn.ParameterList([nn.Parameter(torch.full((1, state_shape), 1.0 + i)) for i in range(self.horizon)])
        self.controls = nn.ParameterList([nn.Parameter(torch.full((1, control_shape), 1.0 + i + self.horizon)) for i in range(self.horizon)])
        self.lambda_ = nn.ParameterList([nn.Parameter(torch.full((1, state_shape), 1.0 + i + self.horizon * 2)) for i in range(self.horizon)])

    def forward(self):
        total_cost = 0
        for k in range(self.horizon):
            states_k = self.x_initial if k == 0 else self.states[k - 1]
            states_kp1 = self.states[k]
            total_cost += (states_kp1 - self.x_final) @ self.Q @ (states_kp1 - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()
            total_cost += self.lambda_[k] @ (states_kp1.t() - self.A @ states_k.t() - self.B @ self.controls[k].t())
        return total_cost * total_cost

state_size = 3
control_size = 2

A = torch.tensor(np.identity(state_size, dtype=np.float32))
B = torch.tensor([[-1, 0], [-1, 0], [0, -1]], dtype=torch.float32)

Q = torch.tensor(np.identity(state_size, dtype=np.float32))
R = torch.tensor(np.identity(control_size, dtype=np.float32))

model = MPC(A=A, B=B, Q=Q, R=R, T=0.2, horizon=100, state_shape=state_size, control_shape=control_size)
optimizer = torch.optim.Adam(model.parameters())

loss_list = []

for i in range(100000):
    optimizer.zero_grad()
    total_cost: torch.Tensor = model()
    total_cost.backward()
    optimizer.step()

    with torch.no_grad():
        loss_list.append(total_cost.detach().cpu().item())
        if len(loss_list) > 1000:
            loss_list = loss_list[1:]
        if i % 100 == 0:
            plt.clf()
            plt.plot(loss_list, label='Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss over iterations')
            plt.legend()
            plt.pause(0.1)