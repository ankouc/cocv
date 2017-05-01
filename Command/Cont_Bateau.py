from Control_1V import *

# Definition of boundary conditions

constraints1 = {'lower': 0.0, 'upper': 10.0}
constraints2 = {'lower': None, 'upper': None}
constraintsf = {'lower':None, 'upper':None}
bcs = {'bc_u1':constraints1, 'bc_f': constraintsf}


# Initial shape of the command

f0 = 'sin(pi*x[0])'


# Definition of objective and state functions

objective = '(inner(self.u1.dx(0),self.u1.dx(0)))**(0.5)*dx'
state_func = '(inner(grad(self.u1), grad(self.v1))-200*self.v1*cos(self.f))'
#state_func = '(dot(self.u1.dx(0), self.v1)-200*self.v1*cos(self.f))'


# Call of the solver

sol = Optimal_command(
                      n_vertice = 200,
                      objective = objective,
                      state_func = state_func,
                      bcs = bcs,
                      f0 = f0,                   
                      solver = 'BFGS')


sol.main()
