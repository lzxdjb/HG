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
        self.T = T
        self.horizon = horizon
        self.state_shape = state_shape
        self.control_shape = control_shape

        self.cost = control_shape
        self.x_initial = x_initial
        self.x_final = x_final

        self.equalityADMM = []
        self.JBADMM = []
        self.HessianADMM = []
        self.JB = []

####### test
        # self.states = [x_initial] + [torch.full((1 ,self.state_shape), 1,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        # self.controls = [torch.full( (1 ,self.control_shape), 1,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]


######
        
###### real number:

        self.states = [x_initial] + [torch.full((1 ,self.state_shape), 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.states)

        self.controls = [torch.full( (1 ,self.control_shape), 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.controls)

        self.lambda_ = [torch.full( (1 ,self.state_shape), 1 ,  dtype = torch.float64) for i in range(self.horizon)]
######
        
    def check_shapes(self , states):
        for i, state in enumerate(states):
            print(f"Shape of state {i}: {state.shape}")
    def bug(self ):
        for i, state in enumerate(self.states):
            print(f"Shape of state {i}: {state}")

        for i, state in enumerate(self.controls):
            print(f"Shape of control {i}: {state}")

       
        # print(f"lamda_ : {self.lambda_}")

    def debug(self):
        # print("jacobian = " , self.jacobian.shape)
        # print("jacobian = " , self.jacobian)
        # print()

        # print("hessian  = " , self.hessian.shape)
        # print("hessian = " , self.hessian)
        # print()

        print("gradient = " , self.Subgradient)
      

        print("self.constrain_h = " , self.constrain_h.shape)
        print("self.constrain_h = " , self.constrain_h)

    def ttttotal_cost(self):
        jj = 0.0
        for k in range(self.horizon):
            jj += (self.states[k] - self.x_final) @ self.Q @ (self.states[k] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()
        # jj += (self.states[0] - self.x_final) @ self.Q @ (self.states[0] - self.x_final).t()
        
        return jj
    
    def constrain(self , i):
        equality = []

        equality.append(self.states[i + 1][0][0] - self.states[i][0][0] - self.T * torch.cos(self.states[i][0][2]) * self.controls[i][0][0])
        equality.append(self.states[i + 1][0][1] - self.states[i][0][1] - self.T * torch.sin(self.states[i][0][2]) * self.controls[i][0][0])
        equality.append(self.states[i + 1][0][2] - self.states[i][0][2] - self.T * self.controls[i][0][1])

        # print("equality = " , equality)

        temp = 0
        for k in range(0 , self.state_shape):
            temp += self.lambda_[i][0][k] * equality[k]

        return temp
    
    def getContrain(self , i):
        equality = []

  
        equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][0] - self.states[i][0][0] - self.T *torch.cos(self.states[i][0][2]) * self.controls[i][0][0]), dim=0))

        equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][1] - self.states[i][0][1] - self.T *torch.sin(self.states[i][0][2]) * self.controls[i][0][0]), dim=0))
            
        equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][2] - self.states[i][0][2] - self.T * self.controls[i][0][1]), dim=0))

        big_tensor = torch.cat(equality, dim=0)
        big_tensor = torch.unsqueeze(big_tensor, dim=1)

        self.constrain_h = big_tensor

        # print("equality = " , big_tensor)
        return big_tensor
    
    def getGradient(self):

        self.GetProblem()
        self.varible = []
        SubProblemVarible = []
        SubProblemVarible.append(self.controls[0])
        self.varible.append(SubProblemVarible)


        for i in range(1 , self.horizon):
            SubProblemVarible = []
            SubProblemVarible.append(self.states[i])
            SubProblemVarible.append(self.controls[i])
        
            self.varible.append(SubProblemVarible)

        SubProblemVarible = []
        SubProblemVarible.append(self.states[self.horizon])
        self.varible.append(SubProblemVarible)
        
        self.Subgradient = []
        # print("self.varible = " , self.varible)
        for i in range(0 , len(self.SubProblem)):
       
            grad_f = torch.autograd.grad(self.SubProblem[i], self.varible[i], create_graph=True, allow_unused=True)
            grad_f = torch.cat(grad_f , dim=1).t()
            self.Subgradient.append(grad_f)

   
    
    def GetProblem(self):
        self.SubProblem = []

  
        self.SubProblem.append(self.controls[0] @ self.R @ self.controls[0].t()  + self.constrain(0))


      
        for i in range(1 , self.horizon):

            self.SubProblem.append( (self.states[i] - self.x_final) @ self.Q @ (self.states[i] - self.x_final).t() + self.controls[i] @ self.R @ self.controls[i].t() + \
            self.constrain(i - 1) + self.constrain(i))

        self.SubProblem.append((self.states[self.horizon] - self.x_final) @ self.Q @ (self.states[self.horizon] - self.x_final).t() + self.constrain(self.horizon - 1))

        # print("problem = " , self.SubProblem)
        

   

    def update(self , learing_rate , iteration):

        # print("self.gradient = " , self.Subgradient)
        temp = self.Subgradient[0].t()
        jj = self.controls[0].clone()
        # print("temp = " , temp.shape)
        jj -= learing_rate * temp
        self.controls[0] = jj

        for i in range(1 , self.horizon):

            temp = self.Subgradient[i].t()
            jj = self.states[i + 1].clone()
            jj -= learing_rate * temp[0 ,  : self.state_shape].unsqueeze(dim = 0)
            self.states[i + 1] = jj

            # print(jj.shape)
            # print(temp[0 , :self.control_shape].unsqueeze(dim = 0).shape)
            
            jj = self.controls[i].clone()
            jj -= learing_rate * temp[0 , self.state_shape 
            : self.state_shape + self.control_shape].unsqueeze(dim = 0)

          
            self.controls[i] = jj

        temp = self.Subgradient[self.horizon].t()
        jj = self.states[self.horizon].clone()
        jj -= learing_rate * temp
        self.states[self.horizon] = jj

        a = 1
        b = 0.1

        for i in range(self.horizon):
            self.lambda_[i] += 1 / (a + b * iteration) * self.getContrain(i).t() / torch.norm(self.getContrain(i))
    
    def updateState(self):
        zero_tensor = torch.zeros_like(self.states[-1])
        shifted_states = self.states[1:] + [zero_tensor]
        self.states = shifted_states

        shifted_control = self.controls[1:] + self.controls[0]
        self.controls = shifted_control

        
        # print(self.states)
        # print(self.controls)

   

    def JudgeFullRank(self ,matrix):

        rank = np.linalg.matrix_rank(matrix.detach().numpy())
        num_rows = matrix.shape[0]
        is_full_row_rank = (rank == num_rows)
        print(f"Rank of the Jacobian matrix: {rank}")
        print(f"Number of rows: {num_rows}")
        print(f"Is the Jacobian matrix full row rank? {'Yes' if is_full_row_rank else 'No'}")


    
    def train(self):

        learning_rate = float(0.01)
        
        iteration = 0
        while 1:
        # for j in range():
            for i in range(self.horizon + 1):

                self.getGradient()
                # subgradient = self.Subgradient[i]
                # print("subgradient = " , subgradient.shape)
                self.update(learning_rate , iteration)

                iteration += 1
                # exit()
            
                lambda_ = torch.stack(self.lambda_)
                loss_current = torch.norm(lambda_)
                cost = self.ttttotal_cost()
                print("############")
                print()
                # self.bug()
                # exit()
                self.debug()
                print("loss = " , loss_current)
                print("cost = " , cost)

                # exit()
                # print()
                # print("############")

                # loss_list.append(cost.item())
                # plt.plot(loss_list, label='Loss')
                # plt.xlabel('Iteration')
                # plt.ylabel('Loss')
                # plt.title('Loss over iterations')
                # plt.legend()
                # plt.grid(True)
                # plt.pause(0.01)  # Add a pause to allow the plot to update
                # plt.clf()

                if loss_current < 1e-8:
                    print("################")
                    states = [tensor.detach().numpy() for tensor in self.states]
                    controls = [tensor.detach().numpy() for tensor in self.controls]
               

                    states = np.concatenate([state for state in states], axis=0)
                    controls = np.concatenate([ctrl for ctrl in controls], axis=0)
                 

                    print("states = " , states.T)
                    print("control = " ,controls.T)
                 
                    jb = self.ttttotal_cost()

                    print("loss = " ,jb )

                    break
                

                
state_size = 3
control_size = 2
T = 1
horizon = 1
x_initial = torch.tensor([[0.0, 0.0 ,0.0]]  , dtype=torch.float64)
# x_initial = torch.tensor([[ 1.0000e+00,  5.4629e-09, -5.0000e-01]]  , dtype=torch.float64)

x_final = torch.tensor([[0, 0 , 0]] , dtype=torch.float64 )
# A = np.identity(state_size)

Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])
# Convert the NumPy arrays to PyTorch tensors
# A = torch.tensor(A, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)


model = MyModel(A=None, B=None  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size) 

model.train()

# model.GetProblem()
# model.getGradient()
# model.constrain(0)
