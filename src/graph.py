import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

class Vertex:
    def __init__(self, id, data=None):
        self.id = id
        self.data = data  # This can be a tensor

    def __str__(self):
        return f"Vertex({self.id}, {self.data})"

    def __repr__(self):
        return self.__str__()


class Edge:
    def __init__(self, start_vertex, end_vertex, weight=1.0):
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
        self.weight = weight  # This can be a scalar tensor

    def __str__(self):
        return f"Edge(from {self.start_vertex} to {self.end_vertex}, weight {self.weight})"

    def __repr__(self):
        return self.__str__()




class NaiveQuestion(Vertex, Edge):
    def __init__(self, id, data, start_vertex, end_vertex, A , B,  Q , R , T , horizon , state_shape , control_shape):
        Vertex.__init__(self, id, data)
        Edge.__init__(self, start_vertex, end_vertex)

        # Setting up additional properties for the optimization problem
        self.time_grid = None
        self.states = None
        self.controls = None
        self.A = A
        # self.B = B
        self.Q = Q
        self.R = R
        self.total_cost = 0
        self.T = T
        self.horizon = horizon
        self.state_shape = state_shape
        self.control_shape = control_shape

       


    def setup_problem(self, x_initial, x_final , upper_state_constrain = None , lower_state_constrain = None ,upper_control_constrain = None , lower_control_constrain = None):
    
        # self.check_shapes(self.time_grid)

        temp = x_initial

        self.states = [temp] + [torch.zeros_like(temp).cuda() for _ in range(self.horizon)] 
        # self.check_shapes(self.states)

        u_init = torch.tensor([[0.0, 0.0 ]], device='cuda' , dtype= torch.float64)

        self.controls = [torch.zeros_like(u_init).cuda() for _ in range(self.horizon)]
        # self.check_shapes(self.controls)

        self.upper_state_constrain= [torch.zeros_like(upper_state_constrain).cuda() for _ in range(self.horizon)]
        # self.check_shapes(self.upper_state_constrain)

        self.lower_state_constrain = lower_state_constrain
        # self.check_shapes(self.lower_state_constrain)

        self.x_initial = x_initial
        self.x_final = x_final

        # self.B = torch.tensor([[ torch.cos(self.x_initial[0][2]), 0], 
        #                        [torch.sin(self.x_initial[0][2]) , 0],
        #                        [0 , 1]] , device='cuda' , dtype=torch.float64)

        self.B = torch.tensor([[-1, 0], 
                               [-1 , 0],
                               [0 , -1]] , device='cuda' , dtype=torch.float64)


    def check_shapes(self , states):
        for i, state in enumerate(states):
            print(f"Shape of state {i}: {state.shape}")

        # print()


    def dynamics(self, x, u):
        """
        Define the system dynamics function f(x, u).
        :param x: State tensor
        :param u: Control tensor
        :return: Dynamics tensor
        """
        # Example dynamics: simple linear system (can be replaced with actual dynamics)
        A = torch.eye(x.size(0)).cuda()  # Identity matrix as placeholder
        B = torch.ones((x.size(0), u.size(0))).cuda()  # Ones matrix as placeholder
        return self.A @ x + self.B @ u

    def cost_function(self, x, u):
        """
        Define the cost function l(x, u).
        :param x: State tensor
        :param u: Control tensor
        :return: Cost tensor
        """
        # Example cost: quadratic cost (can be replaced with actual cost function)
        diff = x - self.x_final
        return  torch.matmul(torch.matmul(diff, self.Q), diff.t())[0][0] #sdasd

    def DirectCollocationCost(self):
        """
        Compute the total cost using Simpson quadrature.
        """
        total_cost = 0.0
        delta_t = self.time_grid[1:] - self.time_grid[:-1]

        total_cost += (self.x_initial - self.x_final).t() @ self.Q @ (self.x_initial - self.x_final)

        for k in range(self.horizon ):
            print("k = " , k)
            x_k = self.states[k]
            x_k1 = self.states[k + 1]
            u_k = self.controls[k]
            u_k1 = self.controls[k + 1]
            u_mid = (u_k + u_k1) / 2
            x_mid = (x_k + x_k1) / 2
            l_k = self.cost_function(x_k, u_k)
            l_mid = self.cost_function(x_mid, u_mid)
            l_k1 = self.cost_function(x_k1, u_k1)
            total_cost += (delta_t[k] / 6) * (l_k + 4 * l_mid + l_k1)
        self.total_cost = total_cost
        return total_cost
    

    def ShootingCost(self):

        total_cost = torch.tensor(0.0).cuda() 

        for k in range(self.horizon):
            # print("k = " , k)
            # print("cost_function = " , self.cost_function(self.states[k] , self.controls[k])[0][0])
            total_cost += self.cost_function(self.states[k] , self.controls[k])
        
        total_cost +=  torch.matmul(torch.matmul(self.states[k+1], self.Q), self.states[k+1].t())[0][0]

        self.total_cost = total_cost * 1 / 2
        return total_cost
    
    
    def getHessian(self,horizon):

        totalShape =  totalShape = horizon * 3 + horizon * 2 + horizon * 3

        hessian = torch.zeros((totalShape  , totalShape), device='cuda' , dtype=torch.float64)
    
        identity = torch.eye(3, device='cuda' , dtype = torch.float64)

        #state
        for i in range(horizon ):
            hessian[3 * i : 3 * (i + 1) , 3 * i : 3 * (i+1)] =2 *  self.Q
            hessian[3 * i : 3 * (i + 1) , 5 * (horizon) + i * 3 : 5 * (horizon) + (i + 1) * 3] = identity

            if i < horizon - 1:
                hessian[3 * i : 3 * (i + 1) , 5 * (horizon) + (i + 1) * 3 : 5 * (horizon) + (i + 2) * 3] = - self.A.t()

        #control
        for i in range(horizon):

            base = 3 * horizon

            hessian[base + 2 * i : base + 2 * (i + 1) , base + 2 * i : base + 2 * (i+1)] = 2 * self.R
            hessian[base + 2 * i : base + 2 * (i + 1) , 5 * (horizon) + i * 3 : 5 * (horizon) + (i + 1) * 3] = - self.B.t()

        #dual
        base = 5 * horizon
        hessian[base + 3 * 0 : base + 3 * (0 + 1) , 0 * 3 : (0 + 1) * 3] = identity
        hessian[base + 3 * 0 : base + 3 * (0 + 1) , 3 * horizon : 3 * horizon + 2] = - self.B

        for i in range(1 , horizon ):

            base = 5 * horizon
            hessian[base + 3 * i : base + 3 * (i + 1) , (i - 1) * 3 : (i) * 3] = - self.A.t()
            hessian[base + 3 * i : base + 3 * (i + 1) , i * 3 : (i + 1) * 3] = identity
            hessian[base + 3 * i : base + 3 * (i + 1) , 3 * horizon + i * 2 : 3 * horizon + (i + 1)* 2] = - self.B



        return hessian
        # print()
    

    def getGradient(self , horizon , vector):

        totalShape = horizon * self.state_shape + horizon * self.control_shape + horizon * self.state_shape


        gradient = torch.zeros((totalShape  , 1), device='cuda' , dtype=torch.float64)

        dual_base = self.state_shape * horizon + self.control_shape * horizon

        # print("Q = " , self.Q)
        # print("self.x_final = " , self.x_final)

        for i in range(horizon):
            # print()
            gradient[i * 3 : (i + 1) * 3] = 2 * self.Q @ (vector[i * self.state_shape] - self.x_final).t() + vector[dual_base + i * 3 : dual_base + 3 * (i+ 1)]

            # print("2 * self.Q @ (vector[i * self.state_shape] - self.x_final).t() " , 2 * self.Q @ (vector[i * self.state_shape] - self.x_final).t() )
            # print("vector[dual_base + i * 3 : dual_base + 3 * (i+ 1) = " , vector[dual_base + i * 3 : dual_base + 3 * (i+ 1)])
            # print("sdsd" , 2 * self.Q @ (vector[i * self.state_shape] - self.x_final).t() )


            if i < horizon - 1:
                gradient[i * 3 : (i + 1) * 3] =gradient[i * 3 : (i + 1) * 3] - self.A.t() @ vector[dual_base + 3 * (i + 1): dual_base + 3 * (i + 2)]

        control_base = horizon * self.state_shape
        # print("gradient = " ,gradient[0 * 3 : (0 + 1) * 3])


        # print("B = " , self.B)
        for i in range(horizon):

            gradient[control_base + i * 2 : control_base + (i + 1) * 2] = 2 * self.R @ vector[control_base + i * self.control_shape : control_base + (i + 1)* self.control_shape] - self.B.t() @ vector[dual_base + 3 * (i) : dual_base + 3 * (i + 1)]

        dual_base = horizon * (self.state_shape + self.control_shape)

        gradient[dual_base + 0 * self.state_shape : dual_base + (0 + 1) * self.state_shape] = vector[0 * self.state_shape : 1 * self.state_shape] - self.A @ self.x_initial.t() - self.B @ vector[control_base + 0 * self.control_shape : control_base + 1 * self.control_shape]

        for i in range(1 , horizon):
            
            a = vector[i * self.state_shape : (i + 1) * self.state_shape]
            b = - self.A @ vector[(i - 1) * self.state_shape : i * self.state_shape] 
            c = - self.B @ vector[control_base + i * self.control_shape : control_base + (i + 1) * self.control_shape]

            # print("a = " , a.shape)
            # print("b = " , b.shape)
            # print("c = " , c.shape)
            # print("i = " , i)
            # print("gradient.shape = " , gradient.shape)
            # print("dual_base = " , dual_base)
            # print("state_shape = " , self.state_shape) 
            # print( gradient[dual_base + i * self.state_shape : dual_base + (i + 1) * self.state_shape].shape)

            gradient[dual_base + i * self.state_shape : dual_base + (i + 1) * self.state_shape] = (a + b + c)

        return gradient
    
    def shift(self , vector):
        control_base = self.state_shape * self.horizon
        dual_base = (self.state_shape + self.control_shape) * self.horizon

        CurrentState = vector[0 + 0 * self.state_shape :  0 + 1 * self.state_shape]
        CurrentControl = vector[control_base + 0 * self.control_shape : control_base + 1 * self.control_shape]

        part1 = vector[:control_base]
        part2 = vector[control_base: dual_base]
        part3 = vector[dual_base:]

        # print("previous = " , part3)

        part1 = torch.cat((part1[3:], part1[:3]), dim=0)
        part2 = torch.cat((part2[2:], part2[:2]), dim=0) 
        part3 = torch.cat((part3[3:], part3[:3]), dim=0)  

        vector = torch.cat((part1 , part2 , part3) )
        # print(vector.shape) 
    
        # print("current = " , part3)
        return CurrentState , CurrentControl , vector

            

    def testParameter(self):
        self.B = torch.tensor([[ -1, 0], 
                               [ - 1 , 0],
                               [0 , -1]] , device='cuda' , dtype=torch.float64)
        vector = torch.tensor([[1,1 , 1 , 2 , 2 , 2 , 3 ,3 , 4 , 4 , 5 , 5 , 5 , 6 , 6 , 6]] , device = 'cuda' , dtype = torch.float64).t()
        self.horizon = 2
            
        return vector

    def Leverberg(self):
        # lambda_ = torch.tensor([[0.5, 0.5, 0.5]], device='cuda' , dtype=torch.float64)
        # loss = 10

        self.horizon = 2

        self.B = torch.tensor([[ torch.cos(self.x_initial[0][2]), 0], 
                               [torch.sin(self.x_initial[0][2]) , 0],
                               [0 , 1]] , device='cuda' , dtype=torch.float64)
        
        self.B = self.B * self.T
        totalShape = 3 * self.horizon + 2 * self.horizon + 3 * self.horizon
        vector = torch.ones((totalShape  , 1), device='cuda' , dtype=torch.float64)

        # vector = self.testParameter()
        # print("horizon = " , self.horizon)
        # print("self.B = " , self.B)
        # print("vector = " , vector.shape)

        # self.horizon = 2

        gradient = torch.zeros((totalShape  , 1), device='cuda' , dtype=torch.float64)
        hessian = self.getHessian(self.horizon)
        # print("hessian = " , hessian)

       
        # print("vctor.shape = " , vector.shape)
      
        # print("asdfasdf = " , vector.shape)

        xx = []
        xx.append(self.x_initial)
        uu = []


        for i in range(20):
        
            while 1:
                # print("vector = " , vector)

                gradient = self.getGradient(self.horizon , vector) 
                loss = torch.norm(hessian.inverse() @ gradient)
                # print("loss  = "  , loss)

                vector = vector - hessian.inverse() @ gradient
                # print("vector  = " , vector)


                if loss < 1e-5:
                    # print("i = " , i)
                    print("final_vector  = " , vector)
                    break

            # exit()

            CurrentState , CurrentControl , vector = self.shift(vector)

            self.x_initial = CurrentState.reshape(1 , 3)
            self.B = torch.tensor([[ torch.cos(self.x_initial[0][2]), 0], 
                               [torch.sin(self.x_initial[0][2]) , 0],
                               [0 , 1]] , device='cuda' , dtype=torch.float64)
            self.B = self.B * self.T
            print("B = " , self.B)

            xx.append(CurrentState.reshape(1 , 3))
            uu.append(CurrentControl)
            # exit()

        print("xx = " , xx)
        return xx

        # for k in range(5):
        #     gradient = self.getGradient(gradient , hessian)

    
    def testLeverberg(self , i ):

        lambda_ = torch.tensor([[0.5, 0.5, 0.5]], device='cuda' , dtype=torch.float64)
        # print("lambda_  =" , lambda_.shape)
        loss = 10

        for k in range(1):

            grad_x1 = 2 * self.Q @ (self.states[i] - self.x_final).t() - self.A.t() @ lambda_.t()
            # print("self.states[i]  =" ,  self.states[i].reshape(3 , 1).shape)
            # print("grad_x1 = " , grad_x1.shape)
            grad_x2 = 2 * self.Q @ (self.states[i + 1]- self.x_final).t() + lambda_.t()
            # print("grad_x2 = " , grad_x2.shape)

            grad_u1 = 2 * self.R @ self.controls[i].t() - self.B.t() @ lambda_.t()
            # print("grad_u1 = " , grad_u1.shape)

            grad_lambda = self.states[i + 1].t() - self.A @ self.states[i].t() - self.B @ self.controls[i].t()
            # print("grad_u1 = " , grad_lambda.shape)

            # Assemble gradients together
            gradient = torch.cat((grad_x1.t(), grad_x2.t(), grad_u1.t(), grad_lambda.t()) , dim = 1)
            gradient = gradient.reshape(-1 , 1)
            # print("gradient = " , gradient.shape)

            H_x1x1 = 2 * self.Q
            H_x2x2 = 2 * self.Q
            H_u1u1 = 2 * self.R
            H_lambdax1 = -self.A
            H_lambdax2 = torch.eye(3, device='cuda' , dtype = torch.float64)
            H_lambdau1 = -self.B

            # The full Hessian matrix
            Hessian = torch.zeros((11, 11), device='cuda' , dtype = torch.float64)
            Hessian[:3, :3] = H_x1x1
            Hessian[3:6, 3:6] = H_x2x2
            Hessian[6:8, 6:8] = H_u1u1
            Hessian[:3, 8:] = H_lambdax1.t()
            Hessian[3:6, 8:] = H_lambdax2.t()
            Hessian[6:8, 8:] = H_lambdau1.t()
            Hessian[8:, :3] = H_lambdax1
            Hessian[8:, 3:6] = H_lambdax2
            Hessian[8:, 6:8] = H_lambdau1

            # print("Hessian = " , Hessian.shape)
            # print("Hessian = " , Hessian)
            # print("controls[i] = " , self.controls[i].shape)

            Unknown =  torch.cat((self.states[i].reshape(3 , 1), self.states[i + 1].reshape(3 , 1),self.   controls[i].reshape(2 , 1) , lambda_.t()))
            # print("Unknown = " , Unknown.shape)
            temp = Unknown

            Unknown = Unknown - Hessian.inverse() @ gradient
            # print("temp = " , temp)
            # print("Unknown = "  , Unknown)

            loss = torch.norm(temp - Unknown)
            # print("loss = " , loss)

            self.states[i] = Unknown[0 : 3].t()
            # print(" self.states[i] =  " ,  self.states[i].shape)

            self.states[i + 1] = Unknown[3 : 6].t()
            # print(" self.states[i] =  " ,  self.states[i + 1].shape)

            self.controls[i] = Unknown[6 : 8].t()
            # print("self.controls[i] = " , self.controls[i].shape)

            lambda_ = Unknown[8 : 11].t()
            # print("lambda_ = " , lambda_)




    


    def SolveMultiShooting(self , learning_rate):

        epoch = 0

        inequaltity_dual_upper = torch.zeros(1, self.state_shape, device='cuda') * self.horizon
        inequaltity_dual_lower = torch.zeros(1, self.state_shape, device='cuda') * self.horizon

        equaltity_dual_state = torch.zeros(1 ,self.state_shape, device='cuda') * self.horizon
        equaltity_dual_control = torch.zeros(1 ,self.control_shape, device='cuda') * self.horizon

        # self.gradient_descent()
       
        
        # print("self.B = " , self.B)
        
        self.B = self.B * self.T

        # print("self.B change = " , self.B)
       

        # print("self.states[1] = " , self.states[1] )
        # print("self.states[2] = " , self.states[2])
        # print("self.controls[1]" , self.controls[1])

        self.testLeverberg(1)

        # print("self.states[1] = " , self.states[1] )
        # print("self.states[2] = " , self.states[2])
        # print("self.controls[1]" , self.controls[1])

        # loss = 10
        # while loss > 1e-1:
        #     for i in range(10):
        #         print()
                


        # while(torch.norm(self.x_initial - self.x_final) < 1e-12 and epoch < 10000):
        #     for i in range(self.horizon):
        #         self.states[i] = self.statess[i] - self.R @ self.states[i]

    def test():
        print()





