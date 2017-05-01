from Control_1V import *

# Definition of boundary conditions

constraints1 = {'lower': 0.0, 'upper': 0.0}
constraints2 = {'lower': None, 'upper': None}
constraintsf = {'lower':None, 'upper':None}
bcs = {'bc_u1':constraints1, 'bc_f': constraintsf}


# Initial shape of the command
	
f0 = 'x[0]'


# Definition of objective and state functions

objective = '(   inner(   self.u1 - (sin(2*pi*x[0]))      , self.u1 - (sin(2*pi*x[0]))  )  )*dx'
state_func1 = '(inner(grad(self.u1),grad(self.v1))-self.f*self.v1)'


# Call of the solver

sol = Optimal_command(n_vertice = 50,
                     f0 = f0,
                     objective = objective,
                     solver='BFGS', 
                     bcs=bcs,
                     state_func = state_func1,)
sol.main()
