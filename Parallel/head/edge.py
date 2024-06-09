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

        self.equality.append(self.state1.state[0][0] - self.state0.state[0][0] - T * (torch.cos(self.state0.state[0][2]) * self.control.control[0][0]))

        self.equality.append(self.state1.state[0][1] - self.state0.state[0][1] - T *(torch.sin(self.state0.state[0][2]) * self.control.control[0][0]))

        self.equality.append(self.state1.state[0][2] - self.state0.state[0][2] -  T * self.control.control[0][1])

      
        
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
            [[-1 , 0 , change[0][2] * torch.sin(change[0][2])], 
            [0 , -1 , -change[0][2] * torch.cos(change[0][2])] , 
            [0  , 0 , -1]] , device = 'cpu' , dtype = torch.float64)