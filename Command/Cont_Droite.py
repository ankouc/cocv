from Control_1V import *

constraints1 = {'lower': 0, 'upper': 2}
constraints2 = {'lower': None, 'upper': None}
constraintsf = {'lower':None, 'upper':None}

f0 = '2*cos(3*x[0])*x[0]'

objective = '(1+self.f**2)**(0.5)*dx'
state_func1 = '(dot(self.u1.dx(0), self.u1) - self.f*self.v1)'
state_func2 = ''
bcs = {'bc_u1':constraints1}


sol = Optimal_command(n_vertice = 50,
                     dim=1,
                     f0 = f0,
                     objective = objective,
                     solver='BFGS', 
                     bcs=bcs,
                     state_func = [state_func1],)
sol.main()
