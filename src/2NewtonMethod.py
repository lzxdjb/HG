import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)


StateShape = 3
ControlShape = 2
x_initial = torch.tensor([[0.0, 0.0 , 0]], device='cuda' , dtype=torch.float64)
x_final = torch.tensor([[1.5, 1.5 , 0]], device='cuda' , dtype=torch.float64)

class Vertex():

    def __init__(self, state):
        self.state = state
        self.gradients = []
        self.jacobians = []

        self.GradientEdge = []
        self.ConstrainEdge = []
        self.StateShape = StateShape
        self.ControlShape = ControlShape

    def add_gradientedge(self, edge):
        self.GradientEdge.append(edge)
    
    def add_constrainedge(self , edge):
        self.ConstrainEdge.append(edge)


 ### directly get?
    def getHessian(self):
        # self.gradients.zero_()
        for edge in self.GradientEdge:
            self.gradients += edge.cost_gradient(self.state)

    def getJacabian(self):
        # self.jacobians.zero_()
        for edge in self.ConstrainEdge:

            tempJacobian = edge.constraint_jacobian(self.state)

            self.jacobians.append(tempJacobian)

    def debug(self):
        # print("states.gradient = " , self.gradients)
        print("states.jacabian = " , self.jacobians)

class GradientEdge():
    
    def __init__(self, state0, control):

        self.state0 = state0
        self.control = control
        self.StateShape = StateShape
        self.ControlShape = ControlShape
     
        if state0:
            state0.add_gradientedge(self)
        if control:
            control.add_gradientedge(self)

    def cost(self):

        if self.state0:
            state0 = self.state0.state
        else:
            state0 = x_initial
        if self.control:
            return state0 @ self.Q @ state0.t() + self.control.state @ self.R @ self.control.state.t()
        else:
            return self.control.state @ self.R @ self.control.state.t()


    def central_difference(self, func, var, epsilon=1e-5):
        grad = torch.zeros_like(var)
        for i in range(var.size(0)):
            var[i] += epsilon
            f_plus = func()
            var[i] -= 2 * epsilon
            f_minus = func()
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
            var[i] += epsilon
        return grad

    def cost_gradient(self):
        
        if self.state0:
            return self.central_difference(self.cost, self.state0)
        else:
            return self.central_difference(self.cost, self.control)
        
class ConstrainEdge():

    def __init__(self , state0, state1, control ):
        self.state0 = state0
        self.state1 = state1
        self.control = control
        self.StateShape = StateShape
        self.ControlShape = ControlShape

        if state0:
            state0.add_constrainedge(self)
        if state1:
            state1.add_constrainedge(self)
        if control:
            control.add_constrainedge(self)

    def constraint(self , i):
        if self.state0:
            state0 = self.state0.state
        else:
            state0 = x_initial
        if self.state1:
            state1 = self.state1.state
        else:
            state1 = torch.zeros((1 , self.StateShape) , device='cuda' , dtype = torch.float64)

        control = self.control.state
        
        if i == 0 or i == 1:

            constraint = state1[i] - state0[i] - torch.cos(state0[2]) * control[0] - torch.sin(state0[2]) * control[1]
        
        if i == 2:
            constraint = state1[i] - state0[i] - control[1]


        return constraint

    def central_difference(self, func, var ,epsilon=1e-5):

        print("var = " , var.shape)

        jacobian = torch.zeros((self.StateShape , var.size(1)) , device='cuda' , dtype = torch.float64)

        grad = torch.zeros_like(var , device='cuda' , dtype = torch.float64)

        for k in range(self.StateShape):
            for i in range(var.size(1)):
                var[0][i] += epsilon
                f_plus = func(i)
                var[0][i] -= 2 * epsilon
                f_minus = func(i)
                grad[0][i] = (f_plus - f_minus) / (2 *  epsilon)
                var[i] += epsilon
            jacobian[k , : ] = grad

        return jacobian

    def constraint_jacobian(self , var):
        return self.central_difference(self.constraint, var)



class NewtonMethod():

    def __init__(self, x_initial, x_final, StateShape, ControlShape, Q, R, A):
        
        self.init = x_initial
        self.final = x_finaStateShape = StateShape
        self.controlShape =ControlShapQ = Q
        self.R = R
        self.A = A
        self.T = T
        self.horizon = horizon


    def debug_output(self):
        print("Vertices and their associated edges:")

        print("Gradient for states")
        for idx, state in enumerate(self.States):
            print(f"States {idx }:")
            for edge in state.GradientEdge:
                print(f"  Edge connecting to state {self.States.index(edge.state0) if edge.state0 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")
        print()
        print("Gradient for control")
        for idx, state in enumerate(self.Controls):
            print(f"Control {idx }:")
            for edge in state.GradientEdge:
                print(f"  Edge connecting to state {self.States.index(edge.state0) if edge.state0 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")
        print()
        print("Jacaobian for state")

        for idx, state in enumerate(self.States):
            print(f"States {idx }:")
            for edge in state.ConstrainEdge:
                print(f"  Edge connecting to state0 {self.States.index(edge.state0) if edge.state0 else 'None'} and state1 {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")

        print()
        print("Jacaobian for control")

        for idx, state in enumerate(self.Controls):
            print(f"Control {idx }:")
            for edge in state.ConstrainEdge:
                print(f"  Edge connecting to state0 {self.States.index(edge.state0) if edge.state0 else 'None'} and state1 {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.control else 'None'}")


        print("\nEdges and their associated vertices:")
        for idx, edge in enumerate(self.GradientEdges):
            print(f"Edge {idx}:")
            print(f"  State 0: {self.States.index(edge.state0) if edge.state0 else 'None'}")
            print(f"  Control: {self.Controls.index(edge.control) if edge.control else 'None'}")

        print("\nEdges and their associated vertices:")
        for idx, edge in enumerate(self.JacabianEdges):
            print(f"Edge {idx}:")
            print(f"  State 0: {self.States.index(edge.state0) if edge.state0 else 'None'}")
            print(f"  State 1: {self.States.index(edge.state1) if edge.state1 else 'None'}")
            print(f"  Control: {self.Controls.index(edge.control) if edge.control else 'None'}")


    def setup_problem(self, x_initial, x_final , upper_state_constrain = None , lower_state_constrain = None ,upper_control_constrain = None , lower_control_constrain = None):

        tempState = torch.tensor([[0,0,0]],device = 'cuda' , dtype = torch.float64)
        tempControl = torch.tensor([[0,0]],device = 'cuda' , dtype = torch.float64)

        self.States = []
        self.Controls = []
        self.GradientEdges = []
        self.JacabianEdges = []

        for _ in range(self.horizon):
            state = Vertex(tempState)
            self.States.append(state)
            control = Vertex(tempControl)
            self.Controls.append(control)
        
        InitEdge = ConstrainEdge( None, self.States[0] , self.Controls[0])
        self.JacabianEdges.append(InitEdge)


        for i in range(0 , self.horizon - 1):
            edge = ConstrainEdge(self.States[i] ,self.States[i + 1] , self.Controls[i + 1])
            self.JacabianEdges.append(edge)
            
            edge = GradientEdge(self.States[i] , self.Controls[i])
            self.GradientEdges.append(edge)

        FinalEdge = GradientEdge(self.States[self.horizon - 1] , self.Controls[self.horizon - 1])

        self.GradientEdges.append(FinalEdge)

    def get_jacabian(self):
        for idx , states in enumerate(self.States):
            states.getJacabian()
            print(f"States {idx }: " ,states.debug())

        for idx , controls in enumerate(self.Controls):
            states.getJacabian()
            controls(f"controls {idx }: " ,states.debug())
    
        

state_size = 3
control_size = 2

A = np.identity(state_size)
B = np.zeros((state_size, control_size))

Q = np.identity(state_size) 

R = np.identity(control_size) # 

# Convert the NumPy arrays to PyTorch tensors
A = torch.tensor(A, dtype=torch.float64)
B = torch.tensor(B, dtype=torch.float64)
Q = torch.tensor(Q, dtype=torch.float64)
R = torch.tensor(R, dtype=torch.float64)

# Move the tensors to the GPU
A = A.cuda()
B = B.cuda()
Q = Q.cuda()
R = R.cuda()

T = 0.1
horizon = 2



nn = NewtonMethod( A , Q , R , T , horizon , state_size , control_size)
nn.setup_problem(x_initial ,x_final)
# nn.debug_output()
nn.get_jacabian()
# nn.getgradient_and_jacabian()
    
