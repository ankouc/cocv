from Control_2V import *

# Definition of boundary conditions

constraints1 = {'lower': 0, 'upper': None}
constraints2 = {'lower': 0, 'upper': None}
constraintsf = {'lower':None, 'upper':None}
bcs = {'bc_u1':constraints1, 'bc_u2':constraints2, 'bc_f':constraintsf}


# Initial shape of the command

f0 = 'cos(pi*x[0])'


# Definition of objective and state functions

objective = '((self.u2-0.0)**2+10*(self.u1-0.5)**2)*ds'
state_func1 = '(dot(self.u1.dx(0),self.v1)-self.v1*(self.f**2)**(0.5)+self.v1*self.u1)'
state_func2 = '(dot(self.u2.dx(0),self.v2)-self.v2*self.u1)'


# Call of the solver

sol = Optimal_command(n_vertice = 50,
                     f0 = f0,
                     objective = objective,
                     solver='BFGS', 
                     bcs=bcs,
                     state_func = [state_func1, state_func2],)
sol.main()
