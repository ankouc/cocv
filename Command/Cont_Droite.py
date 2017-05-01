from Control_1V import *

# Definition of boundary conditions

constraints1 = {'lower': 0.0, 'upper': 2.0}
constraints2 = {'lower': None, 'upper': None}
constraintsf = {'lower':None, 'upper':None}
bcs = {'bc_u1':constraints1, 'bc_f': constraintsf}

# Initial shape of the command

f0 = '2*cos(3*x[0])*x[0]'


# Definition of objective and state functions

objective = '(1+self.u1.dx(0)**2)**(0.5)*dx'

#state_func1 = '((self.u1.dx(0)*self.v1) - self.f*self.v1)'
state_func1  = '(inner(grad(self.u1),grad(self.v1)) - self.f*self.v1)'
state_func2 = ''


# Call of the solver

sol = Optimal_command(n_vertice = 50,
                     f0 = f0,
                     objective = objective,
                     solver='BFGS', 
                     bcs=bcs,
                     state_func = state_func1,)

sol.main()
