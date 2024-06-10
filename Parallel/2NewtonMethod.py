import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

from head.edge import StateCostEdge , StateCostEdge , ControlCostEdge , EqualityEdge

from head.vertex import *
import time
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


    def ProblemGetEquality(self):

        # ProblemEqualityTensorList = []
        # for edge in self.EqualityEdges:
        #     ProblemEqualityTensorList.append(edge.getEquality())
        # self.ProblemEqualityTensor = torch.cat(ProblemEqualityTensorList)
        # print(self.ProblemEqualityTensor)

        for index , state in enumerate(self.States):
            # print("state = " , index)
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

    def GetFinalMatrixInverse(self , StateHessian, ControlHessian , StateJacobian , ControlJacobian):

        # print("StateHessian = " , StateHessian)
        # print("ControlHessian = " , ControlHessian)
        # print("StateJacobian = " , StateJacobian)
        # print("ControlJacobian = " , ControlJacobian)

 
        Total = StateShape + ControlShape
        Hessian = torch.zeros((Total, Total) , device = 'cpu' , dtype = torch.float64)

        Hessian[:StateShape, :StateShape] = StateHessian

        Hessian[StateShape:, StateShape:] = ControlHessian

        # print("Hessian = " , Hessian)

        CombinedJacobian = torch.cat((StateJacobian, ControlJacobian), dim=1)

        # print("CombinedJacobian = " , CombinedJacobian)


        JT = CombinedJacobian.t()
        zero_block = torch.zeros(CombinedJacobian.size(0), CombinedJacobian.size(0) ,dtype=torch.float64)


        top_block = torch.cat((Hessian, JT), dim=1)

        bottom_block = torch.cat((CombinedJacobian, zero_block), dim=1)

        final_matrix = torch.cat((top_block, bottom_block), dim=0)
        # print(final_matrix)
            
        return final_matrix.inverse()
    
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
        for i in range(self.horizon):

            while 1:
            # for i in range():
                
                nn.ProblemGetJB()
                nn.ProblemGetEquality()
                nn.ProblemGetHessian() 
                nn.ProblemGetGradient()

                A = self.GetFinalMatrixInverse(self.States[i].Hessian[0] , self.Controls[i].Hessian[0] , self.States[i].Jacobian[0] , self.Controls[i].Jacobian[0])
                # print("a = " , A)
                B = self.GetFinalColumn(self.States[i].Gradient[0] , self.Controls[i].Gradient[0] , self.States[i].EqualityTensor)

                # exit()

                vector = A @ B

                temp = vector[0: StateShape, 0]
                self.States[i].update(temp.t() , learning_rate)

                temp = vector[StateShape: StateShape + ControlShape , 0]
                self.Controls[i].update(temp.t() , learning_rate)

                loss_current = torch.norm(vector[:StateShape + ControlShape ,0])
                print("loss = " , loss_current)
                # self.debug(i)


                if loss_current < 1e-8:
                    if i != horizon - 1:
                        # print("asd" , self.EqualityEdges[i + 1].state0)

                        self.EqualityEdges[i + 1].state0 = self.States[i].state

                        # print("dkdk" , self.EqualityEdges[i + 1].state0)
                    # exit()
                        # time.sleep(1)
                        print("##############@@@@@@@@@@@@@@")
                    break

        print(self.States)
        print(self.Controls)


                




        

nn = NewtonMethod( A , Q , R , T , horizon , StateShape , ControlShape)
nn.setup_problem(x_initial ,x_final)
# nn.debug_output()
# nn.ProblemGetJB()
# nn.ProblemGetHessian() 
# nn.GetFinalMatrixInverse() 
nn.train() 
