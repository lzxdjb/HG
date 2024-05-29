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


class OptimizeProblem(Vertex, Edge):
    def __init__(self, id, data, start_vertex, end_vertex, A , B, Q , R , weight=1.0 ):
        Vertex.__init__(self, id, data)
        Edge.__init__(self, start_vertex, end_vertex, weight)

        # Setting up additional properties for the optimization problem
        self.time_grid = None
        self.states = None
        self.controls = None
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.total_cost = 0

    def setup_problem(self, t_grid, x_initial, x_final, control_bounds, state_bounds):
        """
        Setup the time grid, initial and final states, and bounds for the optimization problem.
        :param t_grid: Tensor of time grid points
        :param x_initial: Tensor of initial state
        :param x_final: Tensor of final state
        :param control_bounds: Tuple of tensors (u_min, u_max)
        :param state_bounds: Tuple of tensors (x_min, x_max)
        """
        self.time_grid = t_grid
        # self.check_shapes(self.time_grid)

        self.states = [x_initial] + [torch.zeros_like(x_initial).cuda() for _ in range(len(t_grid) - 2)] + [x_final]
        # self.check_shapes(self.states)

        self.controls = [torch.zeros_like(control_bounds[0]).cuda() for _ in range(len(t_grid) )]
        # self.check_shapes(self.controls)

        self.control_bounds = control_bounds
        # self.check_shapes(self.control_bounds)

        self.state_bounds = state_bounds
        # self.check_shapes(self.state_bounds)


    def check_shapes(self , states):
        for i, state in enumerate(states):
            print(f"Shape of state {i}: {state.shape}")

        print()

    def hermite_simpson_collocation(self):
        """
        Implement Hermite-Simpson collocation method for the optimization problem.
        """
        phi_values = []
        delta_t = self.time_grid[1:] - self.time_grid[:-1]
        for k in range(len(delta_t)):
            x_k = self.states[k]
            x_k1 = self.states[k + 1]
            u_k = self.controls[k]
            u_k1 = self.controls[k + 1]
            u_mid = (u_k + u_k1) / 2
            x_mid = 0.5 * (x_k + x_k1) + (delta_t[k] / 8) * (self.dynamics(x_k, u_k) - self.dynamics(x_k1, u_k1))
            phi = (delta_t[k] / 6) * (self.dynamics(x_k, u_k) + 4 * self.dynamics(x_mid, u_mid) + self.dynamics(x_k1, u_k1))
            phi_values.append(phi)
        return phi_values

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
      
        return x.t() @ self.Q @ x + u.t() @ self.R @ u

    def compute_cost(self):
        """
        Compute the total cost using Simpson quadrature.
        """
        total_cost = 0.0
        delta_t = self.time_grid[1:] - self.time_grid[:-1]

        # print("x_k = " , np.array(self.states).shape)
        # print("u_k = " , np.array(self.controls).shape)
        for k in range(len(delta_t) ):
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
    
    def compute_gradient(self):
        optimizer = torch.optim.SGD(params=list(self.states) + list(self.controls), lr=0.001)

        for i in range(0 , 100):
            optimizer.zero_grad()
            self.total_cost.backward(retain_graph=True)
            optimizer.step()
            dx = self.states[0].grad
            print(" dx" , dx)


    def __str__(self):
        return f"OptimizeProblem(Vertex id={self.id}, data={self.data}, " \
               f"Edge from {self.start_vertex} to {self.end_vertex}, weight={self.weight})"

    def __repr__(self):
        return self.__str__()
    


class NaiveQuestion(Vertex, Edge):
    def __init__(self, id, data, start_vertex, end_vertex, A , B,  Q , R , T , horizon):
        Vertex.__init__(self, id, data)
        Edge.__init__(self, start_vertex, end_vertex)

        # Setting up additional properties for the optimization problem
        self.time_grid = None
        self.states = None
        self.controls = None
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.total_cost = 0
        self.T = T
        self.horizon = horizon

    def setup_problem(self, t_grid, x_initial, x_final, control_bounds, state_bounds):
        """
        Setup the time grid, initial and final states, and bounds for the optimization problem.
        :param t_grid: Tensor of time grid points
        :param x_initial: Tensor of initial state
        :param x_final: Tensor of final state
        :param control_bounds: Tuple of tensors (u_min, u_max)
        :param state_bounds: Tuple of tensors (x_min, x_max)
        """
        self.time_grid = t_grid
        # self.check_shapes(self.time_grid)

        self.states = [x_initial] + [torch.zeros_like(x_initial).cuda() for _ in range(self.horizon - 1)] 
        # self.check_shapes(self.states)

        self.controls = [torch.zeros_like(control_bounds[0]).cuda() for _ in range(len(self.horizon))]
        # self.check_shapes(self.controls)

        self.control_bounds = control_bounds
        # self.check_shapes(self.control_bounds)

        self.state_bounds = state_bounds
        # self.check_shapes(self.state_bounds)


    def check_shapes(self , states):
        for i, state in enumerate(states):
            print(f"Shape of state {i}: {state.shape}")

        print()


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
      
        return x.t() @ self.Q @ x + u.t() @ self.R @ u

    def DirectCollocationCost(self):
        """
        Compute the total cost using Simpson quadrature.
        """
        total_cost = 0.0
        delta_t = self.time_grid[1:] - self.time_grid[:-1]

        # print("x_k = " , np.array(self.states).shape)
        # print("u_k = " , np.array(self.controls).shape)
        for k in range(len(delta_t) ):
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
    
    def SingleShootingCost(self):
        total_cost = 0.0
        for k in range(self.horizon):




    def compute_gradient(self):
        optimizer = torch.optim.SGD(params=list(self.states) + list(self.controls), lr=0.001)

        for i in range(0 , 100):
            optimizer.zero_grad()
            self.total_cost.backward(retain_graph=True)
            optimizer.step()
            dx = self.states[0].grad
            print(" dx" , dx)


    def __str__(self):
        return f"OptimizeProblem(Vertex id={self.id}, data={self.data}, " \
               f"Edge from {self.start_vertex} to {self.end_vertex}, weight={self.weight})"

    def __repr__(self):
        return self.__str__()