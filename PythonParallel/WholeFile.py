import torch
import numpy as np
import time
StateShape = 3
ControlShape = 2
x_initial = torch.tensor([[0.0, 0.0 , 0]], device='cpu' , dtype=torch.float64)
x_final = torch.tensor([[1.5, 1.5 , 0]], device='cpu' , dtype=torch.float64)
A = np.identity(StateShape)
B = np.zeros((StateShape, ControlShape))

Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])

# Convert the NumPy arrays to PyTorch tensors
A = torch.tensor(A, dtype=torch.float64)
B = torch.tensor(B, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)

# Move the tensors to the GPU
# A = A.cuda()
# B = B.cuda()
# Q = Q.cuda()
# R = R.cuda()

T = 1
horizon = 1

class StateVertex():

    def __init__(self, state):
        self.state = state

        self.Cost = []
        self.Equality = []

        self.StateShape = StateShape
        self.Jacobian = []
        self.Gradient = []
        self.Hessian = []

        self.EqualityTensorList = []

        self.EqualityTensor = None

    def add_Costedge(self, edge):
        self.Cost.append(edge)
    
    def add_Equalityedge(self , edge):
        self.Equality.append(edge)


    def getEquality(self):
        self.EqualityTensorList = []
        for edge in self.Equality[:1]:
            self.EqualityTensorList.append( edge.getEquality())
        
        self.EqualityTensor = torch.cat(self.EqualityTensorList)
        self.EqualityTensor = self.EqualityTensor.unsqueeze(1)
        # print("EqualityTensor = " ,self.EqualityTensor)

    def getJacobian(self):
        # self.jacobians.zero_()
        self.Jacobian = []
        for index ,edge in enumerate(self.Equality):
            tempJacobian = edge.constraint_jacobian(self.state , index)
            self.Jacobian.append(tempJacobian )
        # print("sdsd = " , self.Jacobian)
    
    def getGradient(self):

        self.Gradient = []
        for index, edge in enumerate(self.Cost):
            tempGradient = edge.CostGradient()
            self.Gradient.append(tempGradient)

  
        # print("sdsd = " , self.Gradient)

    def getHessian(self):
        # self.gradients.zero_()

        self.Hessian = []
        for index, edge in enumerate(self.Cost):
            tempHessian = edge.CostHessian()
            self.Hessian.append(tempHessian)


        # print("sdsd = " , self.Hessian)
            
    def update(self , vector , learning_rate):
        self.state += vector * learning_rate


class ControlVertex():

    def __init__(self, control):
        self.control = control

        self.Cost = []
        self.Equality = []

        self.Jacobian = []

        self.ControlShape = ControlShape

        self.EqualityTensorList = []
        self.Hessian = []
        self.Gradient = []

    def add_Costedge(self, edge):
        self.Cost.append(edge)
    
    def add_Equalityedge(self , edge):
        self.Equality.append(edge)

    # def getEquality(self):
    #     for edge in self.Equality:
    #         self.EqualityTensorList.append( edge.getEquality())
        
    #     self.EqualityTensor = torch.cat(self.EqualityTensorList)
    #     self.EqualityTensor = self.EqualityTensor.unsqueeze(1)
        
    #     print("ControlEqualityTensor = " ,self.EqualityTensor)

 ### directly get?
    def getGradient(self):
        for index, edge in enumerate(self.Cost):
            tempGradient = edge.CostGradient()
            self.Gradient.append(tempGradient)
        # print("sdsd = " , self.Gradient)

    def getHessian(self):
        # self.gradients.zero_()
        for index, edge in enumerate(self.Cost):
            tempHessian = edge.CostHessian()
            self.Hessian.append(tempHessian)
        # print("sdsd = " , self.Hessian)


    def getJacobian(self):
        # self.jacobians.zero_()
        # edge0 = self.Equality[0]
        # tempJacobian = edge0.constraint_jacobian(self.control , 0)
        # self.Jacobian.append(tempJacobian)

        # edge1 = self.Equality[1]
        # tempJacobian = edge1.constraint_jacobian(self.control , 1)
        # self.Jacobian.append(tempJacobian)

        # self.Jacobian.append(tempJacobian)
        for index , edge in enumerate(self.Equality):
            tempJacobian = edge.constraint_jacobian(self.control , index)
            self.Jacobian.append(tempJacobian)
        # print("sdsd = " , self.Jacobian)
   
 

    def update(self , vector , learning_rate):
        self.control += vector


import torch
import numpy as np

from head.vertex import *

class StateCostEdge():
    
    def __init__(self, state0):

        self.state0 = state0
        self.StateShape = StateShape
        self.Q = Q
        state0.add_Costedge(self)
        self.cost = self.state0.state @ self.Q @ self.state0.state.t()
    

    def CostGradient(self):
        return 2 * self.Q @ (self.state0.state - x_final).t()

    def CostHessian(self):
        return 2 * self.Q
        
class ControlCostEdge():
    
    def __init__(self, control):

        self.control = control
        self.StateShape = StateShape
        self.R = R
        control.add_Costedge(self)
        self.cost = self.control.control @ self.R @ self.control.control.t()
    

    def CostGradient(self):
            return 2 * self.R @ (self.control.control).t()

    def CostHessian(self):
        return 2 * self.R
        
class EqualityEdge():

    def __init__(self , x_initial, state1, control ):
        self.state0 = x_initial
        self.state1 = state1
        self.control = control
        self.StateShape = StateShape
        self.ControlShape = ControlShape

        if state1:
            state1.add_Equalityedge(self)
        if control:
            control.add_Equalityedge(self)
            

    def getEquality(self):

        self.equality = []

        # print("self.state1.state[0][0] = " , self.state1.state)
        # print("self.state0[0][0] = " , self.state0)
        # print("control = " , self.control)
        self.equality.append(self.state1.state[0][0] - self.state0[0][0] - T * (torch.cos(self.state0[0][2]) * self.control.control[0][0]))

        self.equality.append(self.state1.state[0][1] - self.state0[0][1] - T *(torch.sin(self.state0[0][2]) * self.control.control[0][0]))

        self.equality.append(self.state1.state[0][2] - self.state0[0][2] -  T * self.control.control[0][1])

      
        
        reshaped_tensors = [t.view(1) for t in self.equality]
        
        equalityTensor = torch.cat(reshaped_tensors )
        self.equalityTensor = equalityTensor
        return equalityTensor


    def constraint_jacobian(self , state , i):
       
        control_cpu = self.control.control.cpu().numpy()
        state_cpu = state.cpu().numpy()
        change = self.state0

        if np.array_equal(state_cpu , control_cpu):
            return torch.tensor(
                [[-torch.cos(change[0][2]) , 0],
                [-torch.sin(change[0][2]) , 0],
                [0 , -1]] , device = 'cpu' , dtype = torch.float64
            )   
        else:
            # print("change[0][2] = " , change[0][0])
            return torch.tensor(
            [[1 , 0 , 0], 
            [0 , 1 , 0], 
            [0  , 0 , 1]], device = 'cpu' , dtype = torch.float64)
class NewtonMethod():

    def __init__(self, x_initial, x_final, StateShape, ControlShape, Q, R, A):
        
        self.init = x_initial
        self.final = StateShape
        self.controlShape = Q
        self.R = R
        self.A = A
        self.T = T
        self.horizon = horizon

        self.sparse = []
        


    def debug_output(self):
        print("Vertices and their associated edges:")

        print("for states")
        for idx, state in enumerate(self.States):
            # print("#######cost :")
            # print(f"States {idx }:")
            # for edge in state.Cost:
            #     print(f"  Edge connecting to state {self.States.index(edge.state0) if edge.state0 else 'None'}")

            print("#######contrain :")
            print(f"States {idx }:")
            for edge in state.Equality:
                print(f"  Edge connecting to  state {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")

        print()
        print("for control")
        for idx, control in enumerate(self.Controls):
            # print("#######cost :")
            # print(f"controls {idx }:")
            # for edge in control.Cost:
            #     print(f"  Edge connecting to state {self.Controls.index(edge.control) if edge.control else 'None'}")

            print("#######contrain :")
            print(f"States {idx }:")
            for edge in control.Equality:
                print(f"  Edge connecting to  state {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")

    def setup_problem(self, x_initial, x_final , upper_state_constrain = None , lower_state_constrain = None ,upper_control_constrain = None , lower_control_constrain = None):

        # tempState = torch.tensor([[1,2,3]],device = 'cpu' , dtype = torch.float64)
        tempControl = torch.tensor([[1,1]],device = 'cpu' , dtype = torch.float64)

        self.States = []
        self.Controls = []

        self.ControlCost = []
        self.StateCost = []

        self.EqualityEdges = []

        for i in range(self.horizon):

            tempState = torch.full((1 , StateShape) , 1, device = 'cpu' , dtype = torch.float64)

            state = StateVertex(tempState)
            self.States.append(state)

            control = ControlVertex(tempControl)
            self.Controls.append(control)
        
     

        for i in range(0 , self.horizon):
            edge = EqualityEdge(x_initial,self.States[i] , self.Controls[i])
            self.EqualityEdges.append(edge)
        
        for j in range(0 , self.horizon):
            edge = StateCostEdge(self.States[j])
            self.StateCost.append(edge)

            edge = ControlCostEdge(self.Controls[j])
            self.ControlCost.append(edge)


    def ProblemGetEquality(self , i):

        self.States[i].getEquality()
      
           
    def ProblemGetJB(self , i):
        
        self.States[i].getJacobian()
        self.Controls[i].getJacobian()

    
    def ProblemGetGradient(self , i):

        self.States[i].getGradient()
        self.Controls[i].getGradient()

    def ProblemGetHessian(self , i):

        self.States[i].getHessian()
        self.Controls[i].getHessian()

    def GetFinalMatrixInverse(self , StateHessian, ControlHessian , StateJacobian , ControlJacobian):

 
        Total = StateShape + ControlShape
        Hessian = torch.zeros((Total, Total) , device = 'cpu' , dtype = torch.float64)

        Hessian[:StateShape, :StateShape] = StateHessian

        Hessian[StateShape:, StateShape:] = ControlHessian
        CombinedJacobian = torch.cat((StateJacobian, ControlJacobian), dim=1)
        JT = CombinedJacobian.t()

        zero_block = torch.zeros(CombinedJacobian.size(0), CombinedJacobian.size(0) ,dtype=torch.float64)
        top_block = torch.cat((Hessian, JT), dim=1)
        bottom_block = torch.cat((CombinedJacobian, zero_block), dim=1)
        final_matrix = torch.cat((top_block, bottom_block), dim=0)
        final_matrix = final_matrix.inverse()
        # print("asdfasdf" , final_matrix.inverse())


        # inverse_hessian = Hessian.inverse()
        # temp = (CombinedJacobian @ inverse_hessian @ JT).inverse()
        # top_block = torch.cat((inverse_hessian - inverse_hessian @ JT @ temp @ CombinedJacobian @ inverse_hessian ,inverse_hessian @ JT @ temp), dim=1)

        # bottom_block = torch.cat((temp @ CombinedJacobian @ inverse_hessian, - temp), dim=1)
        # final_matrix = torch.cat((top_block, bottom_block), dim=0)
        # print("jbjbjbjb" , final_matrix)
        # exit()
            
        return final_matrix
    
    def GetFinalColumn(self , StateGradient , ControlGradient  , StateContrain):
        
        # print("StateGradient = " ,  StateGradient)
        # print("StateGradient" , ControlGradient)
        # print("StateContrain = ",StateContrain)
        final_g = torch.cat((StateGradient ,ControlGradient) , dim = 0 )

        # print("final_g = " , final_g.shape)
        # print("StateContrain = " , StateContrain.shape)

        final_column = torch.cat((final_g , StateContrain) , dim = 0)
        # print("asdfasdf = " , final_column)
        return  - final_column
    
    def debug(self ,  i):
        print("EqualityTensor" , self.States[i].EqualityTensor)
        print("self.States[i]" , self.States[i].state)
        print("self.Controls[i]" , self.Controls[i].control)           
    
    def train(self):


        learning_rate = float(1)

        # for i in range(self.horizon):
        # print(len(self.EqualityEdges))
        # exit()
        start_time = time.time()
        for i in range(self.horizon):

            while 1:
            # for i in range():
                start_time = time.time()
                nn.ProblemGetJB(i)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("ProblemGetJB = " , elapsed_time)

                start_time = time.time()
                nn.ProblemGetEquality(i)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("ProblemGetEquality = " , elapsed_time)
            
                start_time = time.time()
                nn.ProblemGetHessian(i) 
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("ProblemGetHessian = " , elapsed_time)

                start_time = time.time()
                nn.ProblemGetGradient(i)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("ProblemGetGradient = " , elapsed_time)


                start_time = time.time()
                A = self.GetFinalMatrixInverse(self.States[i].Hessian[0] , self.Controls[i].Hessian[0] , self.States[i].Jacobian[0] , self.Controls[i].Jacobian[0])
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("GetFinalMatrixInverse = " , elapsed_time)
            
                # print("a = " , A)
                start_time = time.time()
                B = self.GetFinalColumn(self.States[i].Gradient[0] , self.Controls[i].Gradient[0] , self.States[i].EqualityTensor)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("GetFinalColumn = " , elapsed_time)
                # exit()

                vector = A @ B

                temp = vector[0: StateShape, 0]
                self.States[i].update(temp.t() , learning_rate)

                temp = vector[StateShape: StateShape + ControlShape , 0]
                self.Controls[i].update(temp.t() , learning_rate)

                loss_current = torch.norm(vector[:StateShape + ControlShape ,0])
                # print("loss = " , loss_current)
                # self.debug(i)


                if loss_current < 1e-8:
                    if i != horizon - 1:
                        # print("asd" , self.EqualityEdges[i + 1].state0)

                        self.EqualityEdges[i + 1].state0 = self.States[i].state

                        # print("dkdk" , self.EqualityEdges[i + 1].state0)
                    # exit()
                        # time.sleep(1)
                        # print("##############@@@@@@@@@@@@@@")
                    break

        # print(self.States)
        # print(self.Controls)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("elapsed_time = " , elapsed_time)


                




        

nn = NewtonMethod( A , Q , R , T , horizon , StateShape , ControlShape)
nn.setup_problem(x_initial ,x_final)
# nn.debug_output()
# nn.ProblemGetJB()
# nn.ProblemGetHessian() 
# nn.GetFinalMatrixInverse() 
nn.train() 
