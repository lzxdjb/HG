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
