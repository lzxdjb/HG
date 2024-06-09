import torch
import numpy as np
torch.autograd.set_detect_anomaly(True)

class Vertex:
    def __init__(self, state):
        self.state = state
        self.gradients = torch.zeros_like(state)
        self.jacobians = torch.zeros((state.size(0), state.size(0)), device=state.device, dtype=state.dtype)
        self.edges = []

    def add_edge(self, edge):
        self.edges.append(edge)

    def accumulate_gradients_and_jacobians(self):
        self.gradients.zero_()
        self.jacobians.zero_()
        for edge in self.edges:
            self.gradients += edge.cost_gradient()
            # self.jacobians += edge.constraint_jacobian()

    def debug(self):
        print("states.gradient = " , self.gradients)
        print("states.jacabian = " , self.jacobians)

class Edge:
    def __init__(self, StateShape = None ,ControlShape = None , state0=None, state1=None, control=None, Q=None, R=None, A=None):

        self.StateShape = StateShape
        self.controlShape =ControlShape
        self.state0 = state0
        self.state1 = state1
        self.control = control
        self.Q = Q
        self.R = R
        self.A = A

        if state0:
            state0.add_edge(self)
        if state1:
            state1.add_edge(self)
        if control:
            control.add_edge(self)

    def cost(self):
        if self.state0.state:
            state0 = self.state0.state
        else:
            state0 = torch.zeros((self.StateShape))
        if self.control.state:
            control = self.control.control
        else:
            control = torch.zeros((self.controlShape))

        return state0 @ self.Q @ state0 + control @ self.R @ control

    def constraint(self):
        state0 = self.state0.state
        state1 = self.state1.state
        control = self.control.control
        constraint = state1 - state0 + self.A @ state0 - torch.cos(state0[2]) * control[0] - torch.sin(state0[2]) * control[1]
        return constraint

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
        return self.central_difference(self.cost, self.state0)

    def constraint_jacobian(self):
        return self.central_difference(self.constraint, self.state0.state)





class NewtonMethod(Edge):
    def __init__(self, A ,  Q , R , T , horizon , state_shape , control_shape):

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

    def debug_output(self):
        print("Vertices and their associated edges:")
        for idx, state in enumerate(self.States):
            print(f"States {idx }:")
            for edge in state.edges:
                print(f"  Edge connecting to state {self.States.index(edge.state0) if edge.state0 else 'None'} and state {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.state1 else 'None'}")

        for idx, control in enumerate(self.Controls):
            print(f"Control {idx}:")
            for edge in control.edges:
                print(f"  Edge connecting to state {self.States.index(edge.state0) if edge.state0 else 'None'} and state {self.States.index(edge.state1) if edge.state1 else 'None'} and control{self.Controls.index(edge.control) if edge.state1 else 'None'}")

        print("\nEdges and their associated vertices:")
        for idx, edge in enumerate(self.Edges):
            print(f"Edge {idx}:")
            print(f"  State 0: {self.States.index(edge.state0) if edge.state0 else 'None'}")
            print(f"  State 1: {self.States.index(edge.state1) if edge.state1 else 'None'}")
            print(f"  Control: {self.Controls.index(edge.control) if edge.control else 'None'}")



    def setup_problem(self, x_initial, x_final , upper_state_constrain = None , lower_state_constrain = None ,upper_control_constrain = None , lower_control_constrain = None):

        tempState = torch.tensor([0,0,0],device = 'cuda' , dtype = torch.float64)
        tempControl = torch.tensor([0,0],device = 'cuda' , dtype = torch.float64)

        self.States = []
        self.Controls = []
        self.Edges = []

        for _ in range(self.horizon):
            state = Vertex(tempState)
            self.States.append(state)
            control = Vertex(tempControl)
            self.Controls.append(control)
        
        InitEdge = Edge(self.state_shape , self.control_shape , None , self.States[0],self.Controls[0] , self.Q, self.R , self.A )
        self.Edges.append(InitEdge)

        for i in range(0 , self.horizon - 1):
            edge = Edge(self.state_shape , self.control_shape , self.States[i] ,self. States[i + 1] , self.Controls[i + 1] , self.Q , self.R , self.A )
            self.Edges.append(edge)

        FinalEdge = Edge(self.state_shape , self.control_shape , self.States[self.horizon - 1]  , None , None, self.Q , self.R ,self.A )

        self.Edges.append(FinalEdge)

    def getgradient_and_jacabian(self):
        for states in self.States:
            states.accumulate_gradients_and_jacobians()

        for controls in self.Controls:
            controls.accumulate_gradients_and_jacobians()
        
        for states in self.States:
            states.debug()

        for controls in self.Controls:
            controls.debug()
        
        
        




state_size = 3
control_size = 2

A = np.identity(state_size)
B = np.zeros((state_size, control_size))

Q = np.identity(state_size) #
Q  = Q * 1
# Q[2 , 2] = 100

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

x_initial = torch.tensor([[0.0, 0.0 , 0]], device='cuda' , dtype=torch.float64)
x_final = torch.tensor([[2, 2 , np.pi / 3]], device='cuda' , dtype=torch.float64)

nn = NewtonMethod( A , Q , R , T , horizon , state_size , control_size)
nn.setup_problem(x_initial ,x_final)
nn.debug_output()
# nn.getgradient_and_jacabian()
    
