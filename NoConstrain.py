import torch
import torch.nn as nn
import numpy as np


import matplotlib.pyplot as plt
class MyModel(nn.Module):
    def __init__(self ,  A , B,  Q , R , T , horizon , state_shape , control_shape ):
        super(MyModel, self).__init__()
     

        # Setting up additional properties for the optimization problem
        self.time_grid = None
        self.states = None
        self.controls = None
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.total_cost = 0
        self.T = T
        self.horizon = horizon
        self.state_shape = state_shape
        self.control_shape = control_shape

        self.cost = control_shape


        x_initial = torch.tensor([[0.0, 0.0 ,0.0]]  , dtype=torch.float64)
        x_final = torch.tensor([[2, 2 , 2]] , dtype=torch.float64 )


        self.states = [x_initial] + [torch.full((1 ,self.state_shape), i + 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        self.controls = [torch.full( (1 ,self.control_shape), i + 1 + self.horizon,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
    
        self.lambda_ = [torch.full( (1 ,self.state_shape), i + 1 + self.horizon * 2,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        self.x_initial = x_initial
        self.x_final = x_final
        self.B = torch.tensor([[-1 , 0 ], 
             [ -1, 0] , 
             [0 , -1]] , dtype= torch.float64)
        

        
        # print("state = " , self.states)
        # print("control = " , self.controls)  
        # print("lambda = " , self.lambda_)  
        # print("self.horizon = " , self.horizon)
        # print("self.A = " , self.A )
        # print("self.B = " , self.B)      
        # print("self.Q = " , self.Q)
        # print("self.R = " , self.R)

    def forward(self):
        # Example forward pass that uses the states and controls
        # self.check_shapes(self.time_grid)
        
        total_cost = 0
        
        for k in range(self.horizon):
            # print("k = " , k)
            total_cost += (self.states[k + 1] - self.x_final) @ self.Q @ (self.states[k + 1] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()

            total_cost += self.lambda_[k] @ (self.states[k + 1].t() - self.A @ self.states[k].t() - self.B @ self.controls[k].t())
        
            total_cost = total_cost ** 2
   
        self.total_cost = total_cost
        return total_cost
    
    def train(self):
        # self.total_cost = self.forward()
        # self.total_cost.backward()
        
        # state = []
        # control = []
        # dual = []

        # with torch.no_grad():

        #     for i in range(self.horizon + 1):
        #         state.append(self.states[i].grad)
        #         if i < self.horizon:
        #             control.append(self.controls[i].grad)
        #             dual.append(self.lambda_[i].grad)


        # print("state = " , state)
        # print("control = " , control)
        # print("dual = " , dual)

        loss_list = []

        learning_rate = float(0.0000003)  # how to fix it
        # for i in range(0 , 2):
        loss_previous= 1e10

        while 1:
            self.total_cost = self.forward()
            self.total_cost.backward()

            with torch.no_grad():
                gradient = []

                for i in range(self.horizon):
                    self.states[i + 1] -= learning_rate * self.states[i + 1].grad
                    # loss += torch.norm(self.states[i + 1].grad)
                    temp = self.states[i + 1].grad.clone()
                    gradient.append(temp)
                    # print("temp1 = " , temp)
                    # print("self.states[i + 1] = " , self.states[i + 1]) 
                    self.states[i + 1].grad.zero_()
                    # print("temp2 = " , temp)
                    # print("self22.states[i + 1] = " , self.states[i + 1]) 



                for i in range(self.horizon):
                    self.controls[i] -= learning_rate * self.controls[i ].grad
                    # loss += torch.norm(self.controls[i ].grad)
                    temp = self.controls[i].grad.clone()
                    gradient.append(temp)
                    self.controls[i].grad.zero_()

                for i in range(self.horizon):
                    self.lambda_[i] -= learning_rate * self.lambda_[i].grad
                    # loss += torch.norm( self.lambda_[i].grad)
                    temp = self.lambda_[i].grad.clone()

                    gradient.append(self.lambda_[i].grad)
                    self.lambda_[i].grad.zero_()


                states_flattened = torch.cat([state.view(-1) for state in self.states[1:]])
                controls_flattened = torch.cat([control.view(-1) for control in self.controls])
                # lambda_flattened = torch.cat([lam.view(-1) for lam in self.lambda_])

                # Concatenate all the flattened tensors into a single big vector
                vector = torch.cat([states_flattened, controls_flattened])
                gradient = torch.cat([gg.view(-1) for gg in gradient])

                loss_current = torch.norm(gradient)

                # print()
                # print("vector = " , vector.reshape(1 , -1)) 
                print("loss = " , loss_current)
                # print("gradient = " , gradient.reshape(1 , -1)) 

                loss_list.append(loss_current.item())

            
                plt.plot(loss_list, label='Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Loss over iterations')
                plt.legend()
                plt.grid(True)
                plt.pause(0.001)  # Add a pause to allow the plot to update

                # Clear the plot for the next iteration
                plt.clf()

                if loss_current < 1e-6:
                    print("vector = " , vector)
                    print("self.state = " , self.states)
                    print("self.controls = " , self.controls)
                    break

                # vector = vector.reshape(totalShape , 1)
               

    

state_size = 3
control_size = 2

A = np.identity(state_size)
B = torch.tensor([[1 , 0 ], 
             [ 1, 0] , 
             [0 , 1]])

Q = np.identity(state_size) # 
R = np.identity(control_size) # 

# Convert the NumPy arrays to PyTorch tensors
A = torch.tensor(A, dtype=torch.float64)
B = torch.tensor(B, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)


T = 0.2
horizon = 2

x_initial = torch.tensor([[0.0, 0.0 ,0.0]]  , dtype=torch.float64)
x_final = torch.tensor([[2, 2 , 2]] , dtype=torch.float64 )



model = MyModel(A=A, B=B  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size)  

model.train()

