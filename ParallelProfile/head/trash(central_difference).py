    def JBState1(self , i , epsilon=1e-9):
        if self.state0:
            state0 = self.state0.state
        else:
            state0 = x_initial
        state1 = self.state1.state
        control = self.control.state
        Nstate0 = state0
        Nstate0[0 , i] -= epsilon
        Pstate0 = state0
        Pstate0[0 , i] += epsilon
        if i == 0:    
            N = state1[i] - Nstate0[i] - torch.cos(state0[2]) * control[0]
            P = state1[i] - Pstate0[i] - torch.cos(state0[2]) * control[0]
            
            return (P - N) / (2 * epsilon)


        if i == 1:
            N = state1[i] - Nstate0[i] - torch.cos(state0[2]) * control[0]
            P =  state1[i] - Pstate0[i] - torch.cos(state0[2]) * control[0]

            return (P - N) / (2 * epsilon)

    def central_difference(self, func, state ,epsilon=1e-9):

        var = state
        print("var = " , var.shape)

        jacobian = torch.zeros((self.StateShape , var.size(1)) , device='cuda' , dtype = torch.float64)

        grad = torch.zeros_like(var , device='cuda' , dtype = torch.float64)

        for k in range(self.StateShape):
            for i in range(var.size(1)):
                var[0][i] += epsilon
                f_plus = func(i)
                var[0][i] -= 2 * epsilon
                f_minus = func(i)
                grad[0][i] = (f_plus - f_minus) / (2 * epsilon)
                var[i] += epsilon
            jacobian[k , : ] = grad

        return jacobian

