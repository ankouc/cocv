from Control_1V import *



constraints1 = {'lower': 0, 'upper': 2}
constraints2 = {'lower': None, 'upper': None}
constraintsf = {'lower':None, 'upper':None}

bcs = {'bc_u1':constraints1}


f0 = '-x[0]'

objective = '(inner(grad(self.u1),grad(self.u1))+1)**(0.5)*dx'
state_func = '(inner(grad(self.u1), grad(self.v1))-200*self.v1*cos(self.f))'


sol = Optimal_command(
                      n_vertice = 200,
                      objective = objective,
                      state_func = state_func,
                      bcs = bcs,
                      f0 = f0,                   
                      solver = 'BFGS')


sol.main()
