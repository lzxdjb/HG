import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

from head.edge import StateCostEdge , StateCostEdge , ControlCostEdge , EqualityEdge

from head.vertex import *

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
        
        for j in range(0 , self.horizon - 1):
            edge = StateCostEdge(self.States[j])
            self.StateCost.append(edge)

            edge = ControlCostEdge(self.Controls[j])
            self.ControlCost.append(edge)


        FinalEdge = ControlCostEdge(self.Controls[self.horizon - 1])
        self.ControlCost.append(FinalEdge)

    def ProblemGetEquality(self):

        # ProblemEqualityTensorList = []
        # for edge in self.EqualityEdges:
        #     ProblemEqualityTensorList.append(edge.getEquality())
        # self.ProblemEqualityTensor = torch.cat(ProblemEqualityTensorList)
        # print(self.ProblemEqualityTensor)

        for index , state in enumerate(self.States):
            print("state = " , index)
            state.getEquality()
        # for index , control in enumerate(self.Controls):
        #     print("control = " , index)
        #     control.getEquality()
           
    def ProblemGetJB(self):
        for index , state in enumerate(self.States):
            state.getJacobian()
            # print("state = " , index , "Jb = " ,state.Jacobian)

        for index , control in enumerate(self.Controls):
            control.getJacobian()
            # print("control = " , index , "Jb = " ,control.Jacobian)
    
    def ProblemGetGradient(self):
        for index , state in enumerate(self.States):
            # print("state = " , index)
            state.getGradient()
        for index , control in enumerate(self.Controls):
            # print("control = " , index)
            control.getGradient()
    def ProblemGetHessian(self):
        for index , state in enumerate(self.States):
            # print("state = " , index)
            state.getHessian()
        for index , control in enumerate(self.Controls):
            # print("control = " , index)
            control.getHessian()

    def GetFinalMatrixInverse(self , StateHessian, controlHessian , Jacobian):
        
        HessianInverse = []
        for i in range(horizon - 1):
            HessianInverse.append(self.States[i].Hessian[0].inverse())

        HessianInverse.append(torch.zeros((StateShape , StateShape) ,device='cpu' , dtype = torch.float64))

        for k in range(horizon):

            HessianInverse.append(self.Controls[k].Hessian[0].inverse())

        self.sparse = []

        i = 0

        self.sparse.append(((i , i) ,self.States[i].Jacobian[0] @ self.States[i].Jacobian[0].t()))

        self.sparse.append(((i , i + 1) , self.States[i].Jacobian[0] @ self.States[i].Jacobian[1].t()))

        for i in range(1 , self.horizon - 1):
            self.sparse.append(((i , i - 1) , self.States[i - 1].Jacobian[1] @ self.States[i - 1].Jacobian[0].t() ))

            self.sparse.append(((i , i) , self.States[i - 1].Jacobian[1] @ self.States[i - 1].Jacobian[1].t() + self.States[i].Jacobian[0] @ self.States[i].Jacobian[0].t()))

            self.sparse.append(((i , i + 1) , self.States[i].Jacobian[0] @ self.States[i].Jacobian[1].t()))

        i = self.horizon - 1

        self.sparse.append(((i , i - 1) , self.States[i - 1].Jacobian[1] @ self.States[i - 1].Jacobian[0].t() ))

        self.sparse.append(((i , i) , self.States[i - 1].Jacobian[1] @ self.States[i - 1].Jacobian[1].t() + self.States[i].Jacobian[0] @ self.States[i].Jacobian[0].t()))

        print("sparse = " , self.sparse)

            
        # clonedStateJB = [copy.deepcopy(instance.Jacobian) for instance in self.States]

        # clonedControlJB = [copy.deepcopy(instance.Jacobian) for instance in self.States]

        # print(clonedStateJB)
        # print(clonedControlJB)
            
    
    def train(self):

        for i in range(self.horizon):



        

nn = NewtonMethod( A , Q , R , T , horizon , StateShape , ControlShape)
nn.setup_problem(x_initial ,x_final)
# nn.debug_output()
# nn.ProblemGetJB()
# nn.ProblemGetHessian() 
# nn.GetFinalMatrixInverse()  
