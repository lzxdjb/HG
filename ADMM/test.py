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


####### test
        # self.states = [x_initial] + [torch.full((1 ,self.state_shape), 1,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        # self.controls = [torch.full( (1 ,self.control_shape), 1,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]


######
        
###### real number:

        self.states = [x_initial] + [torch.full((1 ,self.state_shape), 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.states)

        self.controls = [torch.full( (1 ,self.control_shape), 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.controls)

        # self.lambda_ = torch.zeros((self.horizon * self.state_shape , 1) , dtype = torch.float64)
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
        print("jacobian = " , self.jacobian.shape)
        print("jacobian = " , self.jacobian)
        print()

        print("hessian  = " , self.hessian.shape)
        print("hessian = " , self.hessian)
        print()

        print("gradient = " , self.gradient.shape)
        print("gradient = " , self.gradient)
        print()

        print("self.constrain_h = " , self.constrain_h.shape)
        print("self.constrain_h = " , self.constrain_h)

    def ttttotal_cost(self):
        jj = 0.0
        for k in range(self.horizon):
            jj += (self.states[k] - self.x_final) @ self.Q @ (self.states[k] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()
        # jj += (self.states[0] - self.x_final) @ self.Q @ (self.states[0] - self.x_final).t()
        
        return jj

    def constrain(self):
        equality = []

        for i in range(0 , self.horizon):
            equality.append(self.states[i + 1][0][0] - self.states[i][0][0] - self.T * torch.cos(self.states[i][0][2]) * self.controls[i][0][0])
            equality.append(self.states[i + 1][0][1] - self.states[i][0][1] - self.T * torch.sin(self.states[i][0][2]) * self.controls[i][0][0])
            equality.append(self.states[i + 1][0][2] - self.states[i][0][2] - self.T * self.controls[i][0][1])

        return equality
    
    def getJB(self):

        self.equality = self.constrain()
        jacobian = []
        varible = []

        for i in range(0 , self.horizon):
            varible.append(self.states[i + 1])
        for i in range(0 , self.horizon):
            varible.append(self.controls[i])

        for f in self.equality:

            grads = torch.autograd.grad(f, varible, allow_unused=True, create_graph=True)
            grads = [g if g is not None else torch.zeros_like(v) for g, v in zip(grads, varible)]
            jacobian.append(torch.cat([g.view(-1) for g in grads]))

        jacobian_matrix = torch.stack(jacobian)

        self.jacobian = jacobian_matrix

        print("jacobian_matrix = " , jacobian_matrix)
    
        return jacobian_matrix
    
    def getHessian(self ):
        varible = []

        for i in range(0 , self.horizon):
            varible.append(self.states[i + 1])
        for i in range(0 , self.horizon):
            varible.append(self.controls[i])
        
        total_cost = 0

       
        for k in range(self.horizon):
            total_cost += (self.states[k + 1] - self.x_final) @ self.Q @ (self.states[k + 1] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()

       

        grad_f = torch.autograd.grad(total_cost, varible, create_graph=True, allow_unused=True)

    
        grad_f = torch.cat(grad_f , dim=1)
        # print("grad_f = " , grad_f)
        # exit()
        hessian = []

        for g in grad_f[0]:
            # print("g = " , g)
            grad2 = torch.autograd.grad(g, varible, retain_graph=True, allow_unused=True)
            grad2 = [g2 if g2 is not None else torch.zeros_like(v) for g2, v in zip(grad2, varible)]
            # print("grad2 = " , grad2)
            hessian.append(torch.cat(grad2 , dim=1))


        hessian_matrix = torch.stack(hessian)
        hessian_matrix = torch.squeeze(hessian_matrix , dim = 1)

        self.hessian = hessian_matrix


        # print("hessian = " , hessian_matrix)
        return hessian_matrix

    def getGradient(self):
        varible = []

        for i in range(0 , self.horizon):
            varible.append(self.states[i + 1])
        for i in range(0 , self.horizon):
            varible.append(self.controls[i])
        
        total_cost = 0
       

        for k in range(self.horizon):
            total_cost += (self.states[k + 1] - self.x_final) @ self.Q @ (self.states[k + 1] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()

   

        grad_f = torch.autograd.grad(total_cost, varible, create_graph=True, allow_unused=True)
        grad_f = torch.cat(grad_f , dim=1).t()

        self.gradient = grad_f

        # print("gradient = " , self.gradient)
        return grad_f

    def getContrain(self):
        equality = []

        for i in range(0, self.horizon):
    # Append each scalar value as a tensor to the list
            equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][0] - self.states[i][0][0] - self.T *torch.cos(self.states[i][0][2]) * self.controls[i][0][0]), dim=0))

            equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][1] - self.states[i][0][1] - self.T *torch.sin(self.states[i][0][2]) * self.controls[i][0][0]), dim=0))
            
            equality.append(torch.unsqueeze(torch.tensor(self.states[i + 1][0][2] - self.states[i][0][2] - self.T * self.controls[i][0][1]), dim=0))

        big_tensor = torch.cat(equality, dim=0)
        big_tensor = torch.unsqueeze(big_tensor, dim=1)

        self.constrain_h = big_tensor

        print("equality = " , big_tensor)
        return big_tensor

    def update(self , vector , learing_rate):

        control_base = self.horizon * self.state_shape

        dual_base = self.horizon * (self.control_shape + self.state_shape)

        temp = vector[0: control_base, 0]
        for i in range(self.horizon):
            jj = self.states[i + 1].clone()
            jj += learing_rate * temp[i * self.state_shape : (i +1)* self.state_shape].unsqueeze(dim = 0)
            self.states[i + 1] = jj

        temp = vector[control_base: dual_base, 0]
        for i in range(self.horizon):
            jj = self.controls[i].clone()
            jj +=  learing_rate *temp[i * self.control_shape : (i + 1)* self.control_shape].unsqueeze(dim = 0)
            self.controls[i] = jj
        
        # temp = vector[dual_base: , 0]
        # self.lambda_ += learing_rate * temp.unsqueeze(dim = 1)

    def getfinalmatrix(self):
            
        hessian = self.getHessian()
        # print("hessian = ")
        Jacobian = self.getJB()
        JT = Jacobian.t()
        zero_block = torch.zeros(Jacobian.size(0), Jacobian.size(0) ,dtype=torch.float64)

        top_block = torch.cat((hessian, JT), dim=1)

        bottom_block = torch.cat((Jacobian, zero_block), dim=1)

        final_matrix = torch.cat((top_block, bottom_block), dim=0)

        # print(final_matrix)
            
        return final_matrix.inverse()
    
    def getfinalcolumn(self):
        g = self.getGradient()
        h = self.getContrain()

        final_column = torch.cat((g , h) , dim = 0)
        # print("asdfasdf = " , final_column)
        return  - final_column
    
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

        loss_list = []
        totolsize = self.horizon * (self.state_shape  * 2 + self.control_shape)

    
        learning_rate = float(0.2)

        vector = torch.zeros((totolsize , 1) , dtype = torch.float64)
        

        # self.update(vector , learning_rate)
     
        for i in range(1):

            while 1:
            # for i in range(1):
                A = self.getfinalmatrix()
                # exit()
                print("A = " , A)
                # djb = A.clone()
                # det = np.linalg.det(djb.detach())
                # print("invertible ? " , det != 0)

                B = self.getfinalcolumn()
                print("B = " , B)

                # exit()

                # print()
                # self.debug()
                # print()

                vector = A @ B
                print("vector = " , vector)
                # exit()
                self.update(vector ,learning_rate)
                

                loss_current = torch.norm(vector[:self.horizon* (self.state_shape + self.control_shape) ,0])
                cost = self.ttttotal_cost()
                print("############")
                print()
                self.bug()
                self.debug()
                print("loss = " , loss_current)
                # print("cost = " ,cost )
                # exit()
                print()
                print("############")

                # loss_list.append(loss_current.item())
                # plt.plot(loss_list, label='Loss')
                # plt.xlabel('Iteration')
                # plt.ylabel('Loss')
                # plt.title('Loss over iterations')
                # plt.legend()
                # plt.grid(True)
                # plt.pause(0.01)  # Add a pause to allow the plot to update
                # plt.clf()

                loss_list.append(cost.item())
                plt.plot(loss_list, label='Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Loss over iterations')
                plt.legend()
                plt.grid(True)
                plt.pause(0.01)  # Add a pause to allow the plot to update
                plt.clf()

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
x_initial = torch.tensor([[ 1.0000e+00,  5.4629e-09, -5.0000e-01]]  , dtype=torch.float64)

x_final = torch.tensor([[1.5, 1.5 , 0]] , dtype=torch.float64 )
# A = np.identity(state_size)

Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])
# Convert the NumPy arrays to PyTorch tensors
# A = torch.tensor(A, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)


model = MyModel(A=None, B=None  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size) 

# for i in range(2):
# Jb1 = model.getJB()
# Jb2 = model.getJB().t()

# JB = Jb1 @ Jb2
# print("Jb = " , JB)
# model.getJB()
# model.getGradient()
# model.getContrain()
# model.getHessian()
# model.debug()
# exit()
model.train()