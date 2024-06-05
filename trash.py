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

    def forward1(self):
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

model = MPC(A=A, B=B, Q=Q, R=R, T=0.2, horizon=1, state_shape=state_size, control_shape=control_size)
optimizer = torch.optim.Adam(model.parameters())

loss_list = []

learning_rate = 0.001
gradient = []

print("model.states" , model.states)
exit()
for i in range(2):

    print("before = " , model.states[1])
    optimizer.zero_grad()
    print("after = " , model.states[1])

    total_cost: torch.Tensor = model.forward1()
    # print(("cost = " , total_cost))
    # exit()
    total_cost.backward()
    print("total_cost = " , total_cost)
    for i in range(model.horizon):
        model.states[i + 1] -= learning_rate * model.states[i + 1].grad
        # loss += torch.norm(model.states[i + 1].grad)
        temp = model.states[i + 1].grad.clone()
        gradient.append(temp)
        # print("temp1 = " , temp)
        # print("model.states[i + 1] = " , model.states[i + 1]) 
        model.states[i + 1].grad.zero_()
        # print("temp2 = " , temp)

    for i in range(model.horizon):
        model.controls[i] -= learning_rate * model.controls[i ].grad
        # loss += torch.norm(model.controls[i ].grad)
        temp = model.controls[i].grad.clone()
        gradient.append(temp)
        model.controls[i].grad.zero_()       
    for i in range(model.horizon):
        model.lambda_[i] -= learning_rate * model.lambda_[i].grad
        # loss += torch.norm( model.lambda_[i].grad)
        temp = model.lambda_[i].grad.clone()     
        gradient.append(model.lambda_[i].grad)
        model.lambda_[i].grad.zero_()
    # optimizer.step()

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