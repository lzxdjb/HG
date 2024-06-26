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


        state1 = 2
        state2 = 2
        state3 = torch.inf
        control1 = 0.6
        control2 = torch.pi / 4.0

        su = [state1 , state2 , state3]
        sl = [-state1 , -state2 , -state3]
        cu = [control1 , control2]
        cl = [-control1 , -control2]

        self.stateUpper = su
        self.stateLower = sl
        self.controlUpper = cu
        self.controlLower = cl

        self.radium = 0.3

        obs1 = torch.tensor([[0.1 , 0.1 , 0.3]] , dtype = torch.float64)

        self.obs = [obs1]
        self.ClosestObs = []

        self.LowerTolerace = 0.3

####### test
        # self.states = [x_initial] + [torch.full((1 ,self.state_shape), i + 1 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        # self.controls = [torch.full( (1 ,self.control_shape), i + 1 + self.horizon,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]

        # self.lambda_ = torch.zeros((self.horizon * self.state_shape , 1) , dtype = torch.float64)
######
        
###### real number:

        self.states = [x_initial] + [torch.full((1 ,self.state_shape), 0 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.states)

        self.controls = [torch.full( (1 ,self.control_shape), 0 ,  dtype = torch.float64 ,  requires_grad=True) for i in range(self.horizon)]
        # self.check_shapes(self.controls)

        self.lambda_ = torch.zeros((self.horizon * self.state_shape , 1) , dtype = torch.float64)
        # self.check_shapes(self.lambda_)
######
        
    def check_shapes(self , states):
        for i, state in enumerate(states):
            print(f"Shape of state {i}: {state.shape}")

    def bug(self ):
        for i, state in enumerate(self.states):
            print(f"Shape of state {i}: {state}")

        for i, state in enumerate(self.controls):
            print(f"Shape of control {i}: {state}")

    def debug(self):
        # print("jacobian = " , self.jacobian.shape)
        # print("jacobian = " , self.jacobian)
        # print()

        # print("hessian  = " , self.hessian.shape)
        # print("hessian = " , self.hessian)
        # print()

        # print("gradient = " , self.gradient.shape)
        # print("gradient = " , self.gradient)
        # print()

        # print("self.constrain_h = " , self.constrain_h.shape)
        # print("self.constrain_h = " , self.constrain_h)


        # print("self.constrain_h = " , self.constrain_h.shape)
        print("self.constrain_h = " , self.constrain_h)

        print("self.inequlity = " , self.barrier_function())

    def calculate_distance(self, state, obstacle):
        # Calculate Euclidean distance between state and obstacle position (ignoring the radius)
        # print(torch.norm(state[: , :2] - obstacle[ : , :2]))
        return torch.norm(state[: , :2] - obstacle[ : , :2])

    def find_closest_obstacles(self):
        closest_obstacles_indices = []
        for state in self.states[1:]:
            distances = torch.tensor([self.calculate_distance(state, obs) for obs in self.obs], dtype=torch.float64)
            closest_obstacle_index = torch.argmin(distances).item()
            closest_obstacles_indices.append(closest_obstacle_index)

        self.ClosestObs = closest_obstacles_indices
        return closest_obstacles_indices

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
    
        return jacobian_matrix
    
    def getHessian(self):
        varible = []

        for i in range(0 , self.horizon):
            varible.append(self.states[i + 1])
        for i in range(0 , self.horizon):
            varible.append(self.controls[i])
        
        total_cost = 0

        temp = torch.zeros((self.state_shape , self.state_shape) , dtype=torch.float64)
                
        for k in range(self.horizon):
            total_cost += (self.states[k] - self.x_final) @ self.Q @ (self.states[k] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()

        total_cost += (self.states[self.horizon] - self.x_final) @ temp @(self.states[self.horizon] - self.x_final).t()

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
        temp = torch.zeros((self.state_shape , self.state_shape) , dtype=torch.float64)

        for k in range(self.horizon):
            total_cost += (self.states[k] - self.x_final) @ self.Q @ (self.states[k] - self.x_final).t() + self.controls[k] @ self.R @ self.controls[k].t()

        total_cost += (self.states[self.horizon] - self.x_final) @ temp @(self.states[self.horizon] - self.x_final).t()

        grad_f = torch.autograd.grad(total_cost, varible, create_graph=True, allow_unused=True)
        grad_f = torch.cat(grad_f , dim=1).t()

        self.gradient = grad_f

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

        # print(big_tensor.shape)
        return big_tensor

    def barrier_function(self):
    # Define the inequality constraints h_i(x)
        h_constraints = []
        i = 0
        for index , state in enumerate(self.states[1:]):
            # print("i = " , i)
            i += 1
            h_constraints.append(state[0][0] - self.stateUpper[0])  

            # print("i = " , i)
            i += 1
            h_constraints.append(self.stateLower[0] - state[0][0]) 

            # print("i = " , i)
            i += 1
            h_constraints.append(state[0][1] - self.stateUpper[1])  

            # print("i = " , i)
            i += 1
            h_constraints.append(self.stateLower[1] - state[0][1])  

            # print("i = " , i)
            i += 1
            # print("@@@@@@@@@@" , state[0][2] ,self.stateUpper[2])
            h_constraints.append(state[0][2] - self.stateUpper[2])  

            # print("i = " , i)
            i += 1
            h_constraints.append(self.stateLower[2] - state[0][2])

            distance = self.calculate_distance(state , self.obs[self.ClosestObs[index]])
            min_distance = self.obs[self.ClosestObs[index]][0][2] + self.radium

            h_constraints.append(min_distance + self.LowerTolerace - distance)  

        # print("######")  

        for control in self.controls:
            # print("i = " , i)
            i += 1
            h_constraints.append(control[0][0] - self.controlUpper[0]) 

            # print("i = " , i)
            i += 1
            h_constraints.append(self.controlLower[0] - control[0][0]) 

            # print("i = " , i)
            i += 1
            h_constraints.append(control[0][1] - self.controlUpper[1]) 

            # print("i = " , i)
            i += 1
            h_constraints.append(self.controlLower[1] - control[0][1])  
            # print("^^^^^^^^^^")

        # Compute the barrier function phi(x)
        barrier = 0
        for index , h in enumerate(h_constraints):
            
            barrier -= torch.log(-h)
            # print("index = " , index ," h = " , h ,  "value = " , torch.log(-h))
        return barrier
    
    
    def getBarrierHessianAndGradient(self):

        variables = []
        for i in range(self.horizon):
            variables.append(self.states[i + 1])
        for i in range(self.horizon):
            variables.append(self.controls[i])


        # Compute the barrier function
        self.barrier = self.barrier_function()
        
        grad_f = torch.autograd.grad(self.barrier, variables,create_graph=True,   allow_unused=True)
        grad_f = torch.cat(grad_f, dim=1)

        self.InEqualityGradint = grad_f
        # print("frad_f = " , grad_f)
        # exit()
        hessian = []

        for g in grad_f[0]:
            grad2 = torch.autograd.grad(g,  variables, retain_graph=True,    allow_unused=True)
            grad2 = [g2 if g2 is not None else  torch.zeros_like(v) for g2, v in zip (grad2, variables)]
            hessian.append(torch.cat(grad2, dim=1))

        hessian_matrix = torch.stack(hessian)
        hessian_matrix = torch.squeeze  (hessian_matrix, dim=1)

        self.InEqualityHessian = hessian_matrix
     
        # exit()
        return grad_f.t() , hessian_matrix 

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

    def getfinalmatrix(self , alpha):
            
        hessian = self.getHessian() * alpha
        _ , inequaltiyHessian = self.getBarrierHessianAndGradient()
        hessian += inequaltiyHessian

        Jacobian = self.getJB()

        JT = Jacobian.t()
        zero_block = torch.zeros(Jacobian.size(0), Jacobian.size(0) ,dtype=torch.float64)

        top_block = torch.cat((hessian, JT), dim=1)

        bottom_block = torch.cat((Jacobian, zero_block), dim=1)

        final_matrix = torch.cat((top_block, bottom_block), dim=0)
            
        return final_matrix.inverse()
    
    def getfinalcolumn(self , alpha):

        g = self.getGradient() * alpha
        inequalityQ , _ = self.getBarrierHessianAndGradient()
        g = g + inequalityQ

        h = self.getContrain()

        final_column = torch.cat((g , h) , dim = 0)
        # print("asdfasdf = " , final_column.shape)
        return  - final_column
    
    def updateState(self):
        zero_tensor = torch.zeros_like(self.states[-1])
        shifted_states = self.states[1:] + [zero_tensor]
        self.states = shifted_states

        shifted_control = self.controls[1:] + self.controls[0]
        self.controls = shifted_control

        self.lambda_.zero_()
        
        print(self.states)
        print(self.controls)
        # print(self.lambda_)

    def JudgeFullRank(self ,matrix):

        rank = np.linalg.matrix_rank(matrix.detach().numpy())
        num_rows = matrix.shape[0]
        is_full_row_rank = (rank == num_rows)
        print(f"Rank of the Jacobian matrix: {rank}")
        print(f"Number of rows: {num_rows}")
        print(f"Is the Jacobian matrix full row rank? {'Yes' if is_full_row_rank else 'No'}")

    def outputResult(self):
        print("################")
        states = [tensor.detach().numpy() for tensor in self.states]
        controls = [tensor.detach().numpy() for tensor in self.controls]
        lambda_ = [tensor.detach().numpy() for tensor in self.lambda_]

        states = np.concatenate([state for state in states], axis=0)
        controls = np.concatenate([ctrl for ctrl in controls], axis=0)
        lambda_ = np.concatenate([lam for lam in lambda_], axis=0)

        print("states = " , states.T)
        print("control = " ,controls.T)
        # print("lambda = " , lambda_.T)
        jb = self.ttttotal_cost()

        print("cost = " ,jb )

        #################


    
    def train(self):

        loss_list = []
        totolsize = self.horizon * (self.state_shape  * 2 + self.control_shape)

    
        learning_rate = float(0.6)

        vector = torch.zeros((totolsize , 1) , dtype = torch.float64)
        

        self.update(vector , learning_rate)
        alpha = 5

        self.find_closest_obstacles()
        # for i in range(5):
        while 1:

            while 1:
            # for i in range(5):
                # print("1")
                A = self.getfinalmatrix(alpha)
                # djb = A.clone()
                # det = np.linalg.det(djb.detach())
                # print("invertible ? " , det != 0)
                # print("2")

                B = self.getfinalcolumn(alpha)

                # print()
                # self.debug()
                # print()
                print("3")

                vector = A @ B
                self.update(vector ,learning_rate)
                

                loss_current = torch.norm(vector[:self.horizon* (self.state_shape + self.control_shape) ,0])
                # cost = self.ttttotal_cost()

                print("############")
                print()
                self.bug()
                self.debug()
                print("loss = " , loss_current)
                # print("cost = " ,cost )
                print("dual_gap = " , (self.state_shape + self.control_shape) * self.horizon / alpha)
                print()
                print("############")


                loss_list.append(loss_current.item())
                plt.plot(loss_list, label='Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.title('Loss over iterations')
                plt.legend()
                plt.grid(True)
                plt.pause(0.01)  # Add a pause to allow the plot to update
                plt.clf()


                if loss_current < 1e-8:

                    self.outputResult()
                    alpha = alpha * 2
                    break

            if (self.state_shape + self.control_shape) * self.horizon / alpha < 1e-9:
                    # 
                print("$$$$$$$$$")
                self.outputResult()
                print("alpha = " , alpha)
                break
                
                
state_size = 3
control_size = 2
T = 0.2
horizon = 3
x_initial = torch.tensor([[0.0, 0.0 ,0.0]]  , dtype=torch.float64)
# x_initial = torch.tensor([[1.0, 1.0 ,1.0]]  , dtype=torch.float64)

x_final = torch.tensor([[1.5, 1.5 , 0]] , dtype=torch.float64 )
# A = np.identity(state_size)

Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])
# Convert the NumPy arrays to PyTorch tensors
# A = torch.tensor(A, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)

# state1 = 2
# state2 = 2
# state3 = 1e20
# control1 = 0.6
# control2 = torch.pi / 4

# stateUpper = [state1 , state2 , state3]
# stateLower = [-state1 , -state2 , -state3]
# controlUpper = [control1 , control2]
# controlLower = [-control1 , -control2]

model = MyModel(A=None, B=None  , Q = Q , R = R, T = T , horizon = horizon , state_shape=state_size , control_shape=control_size) 

# model.find_closest_obstacles()
# model.barrier_function()
# model.getBarrierHessianAndGradient()
model.train()