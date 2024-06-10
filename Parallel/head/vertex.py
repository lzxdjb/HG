import torch
import numpy as np

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
horizon = 2

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