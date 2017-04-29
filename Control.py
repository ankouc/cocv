from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import moola
set_log_level(ERROR)
import numpy as np
import matplotlib.pyplot as plt



def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 0.0, tol)
				
def boundary_U(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 1.0, tol)


class Optimal_command(object):

    def __init__(self, n_vertice, objective, state_func, solver):
        self.n = n_vertice
        self.objective = objective
        self.state_func = state_func
        #self.constraints = constraints
        self.solver = solver


    def make_function_spaces(self, dim=1):
        if dim ==1:
            self.mesh_state = UnitIntervalMesh(self.n)
            self.mesh_control = UnitIntervalMesh(self.n)
        elif dim ==2:
            self.mesh_state = UnitSquareMesh(self.n, self.n)
            self.mesh_control = UnitSquareMesh(self.n, self.n)
        else:
            assert('Dimension > 2 are not implemented yet')


    def extract_values(self, n,m):
        self.x_array = np.linspace(0, 1, n+1)	
        self.f_array = np.reshape(self.f.compute_vertex_values(), [n+1, m+1])
        self.f_opt_array = np.reshape(self.f_opt.compute_vertex_values(), [n+1, m+1])
        self.u_array = np.reshape(self.u.compute_vertex_values(), [n+1, m+1])
      			
    def set_boundaries(self):
        
        target_L = Constant(0.0)
        target_U = Constant(6.0)

        bc_L = DirichletBC(self.V, target_L, boundary_L)
        bc_U = DirichletBC(self.V, target_U, boundary_U)
        self.bc = [bc_L, bc_U]

    def solve(self):
        self.make_function_spaces(1)

        self.V = FunctionSpace(self.mesh_state, "CG", 1)
        W = FunctionSpace(self.mesh_state, "DG", 0)

        self.f = interpolate(Expression("x[0]",degree=1), W, name='Control')        
#self.f = interpolate(Expression("sin(3.141592*x[0])*x[0]",degree=1), W, name='Control')
        self.u = Function(self.V, name='State')
        v = TestFunction(self.V)


        # State equations------------------------------------------------------------------------------------------------------------------
        F = (eval(self.state_func))*dx


        self.set_boundaries()
        solve(F == 0, self.u, bcs=self.bc)


        x = SpatialCoordinate(self.mesh_state)

        J = Functional(eval(self.objective))
        control = Control(self.f)

        rf = ReducedFunctional(J, control)


        problem = MoolaOptimizationProblem(rf)
        f_moola = moola.DolfinPrimalVector(self.f)
        if self.solver == 'CG':
            solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-1,
                                                               'maxiter': 20,
                                                               'display': 3,
                                                               'ncg_hesstol': 0})
        elif self.solver == 'BFGS':
            solver = moola.BFGS(problem, f_moola, options={'gtol': 1E-15,
                                                           'Hinit': "default",
                                                           'jtol': 1E-15,
                                                           'maxiter': 15})
        else:
            pass   	  																																														

        sol = solver.solve()


        self.f_opt = sol['control'].data

        F = (inner(self.u.dx(0), v.dx(0)))*dx
        solve(F == 0, self.u, bcs= self.bc)


        self.extract_values(self.n,0)
    
        self.make_plots()

    def make_plots(self):

        plt.subplot(3,1,1)
        plt.plot(self.x_array, self.f_array[:,0])
        plt.title('Intial command')
        plt.grid()


        plt.subplot(3,1,2)
        plt.plot(self.x_array, self.f_opt_array[:,0])
        plt.title('Optimal command')
        plt.grid()

        plt.subplot(3,1,3)
        plt.plot(self.x_array, self.u_array[:,0])
        plt.title('Optimal state')
        plt.xlabel('T variable')
        plt.grid()
        plt.show()


sol = Optimal_command(n_vertice = 2000,
                      objective = '(1+self.f**2)**(0.5)*dx',
                      state_func = '(inner(grad(self.u), grad(v))-v*self.f)',                      
#state_func = '(inner(grad(self.u), grad(v))-v*self.f)',
                      solver = 'BFGS')


sol.solve()



