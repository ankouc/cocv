from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import moola
set_log_level(ERROR)
import numpy as np
import matplotlib.pyplot as plt


def boundary1_U(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 1.0, tol)

def boundary1_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 0.0, tol)


class Optimal_command(object):

    def __init__(self, state_func, solver, bcs, objective, f0, n_vertice=20):
        self.bc_u1 = bcs['bc_u1']
        self.f0 = f0
        self.state_func1 = state_func[0]
        self.n = 50
        self.objective = objective
        self.solver = solver


    def make_function_space(self):
        mesh_state = UnitIntervalMesh(self.n)
        self.W = FunctionSpace(mesh_state, "DG", 0)
        self.V1 = FunctionSpace(mesh_state, "CG", 1)


        x = SpatialCoordinate(mesh_state)

        self.f = interpolate(Expression(self.f0, degree=1), self.W, name='Control')
        self.u1 = Function(self.V1, name='State1')
        self.v1 = TestFunction(self.V1)
        self.w = TestFunction(self.W)


    def make_boundaries(self, space):
        
        self.bc1 = []


        if self.bc_u1['lower'] is not None:
            bc1_l = DirichletBC(space[0], self.bc_u1['lower'], boundary1_L)         
            self.bc1.append(bc1_l)

        if self.bc_u1['upper'] is not None:
            bc1_u = DirichletBC(space[0], self.bc_u1['upper'], boundary1_U)
            self.bc1.append(bc1_u)
 


    def solve_pb(self, problem):

        f_moola = moola.DolfinPrimalVector(self.f)

        if self.solver == 'CG':
            solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1E-9,
                                                               'maxiter': 20,
                                                               'display': 3,
                                                               'ncg_hesstol': 0})
        elif self.solver == 'BFGS':
            solver = moola.BFGS(problem, f_moola, options={'gtol': 1E-15,
                                                           'Hinit': "default",
                                                           'jtol': 1E-15,
                                                           'maxiter': 150})
        else:
            raise NameError('No other solver implemented yet!') 


        solution = solver.solve()
        return solution 

    def condition_string(self):

        if self.bc1 is not None:
            self.str_cond1 = 'self.bc1'
        else:
            self.str_cond1 = ''


    def main(self):

	
        self.make_function_space()

        self.make_boundaries([self.V1, self.W])

        self.condition_string()

        F = (inner(grad(self.u1), grad(self.v1))-self.v1*self.f)*dx
        target = Expression("x[0]",degree=1)

        solve((inner(grad(self.u1), grad(self.v1))-self.v1*self.f)*dx == 0, self.u1, eval(self.str_cond1))

        J = Functional((1+self.f**2)**(0.5)*dx)
        control = Control(self.f)

        rf = ReducedFunctional(J, control)
        problem = MoolaOptimizationProblem(rf)
        
        solution = self.solve_pb(problem)



        self.f_opt = solution['control'].data
        F_opt = (inner(grad(self.u1), grad(self.v1))-self.v1*self.f_opt)*dx

        solve(F_opt == 0, self.u1, eval(self.str_cond1))

        self.extract_values()
        self.make_plots()










    def extract_values(self):
        self.x_array = np.linspace(0, 1, self.n+1)	
        self.f_array = self.f.compute_vertex_values()
        self.f_opt_array = self.f_opt.compute_vertex_values()
        self.u1_array = self.u1.compute_vertex_values()

    def make_plots(self):
        plt.subplot(3,1,1)
        plt.plot(self.x_array, self.f_array)
        plt.title('Intial command')
        plt.grid()


        plt.subplot(3,1,2)
        plt.plot(self.x_array, self.f_opt_array)
        plt.title('Optimal command')
        plt.grid()

        plt.subplot(3,1,3)
        plt.plot(self.x_array, self.u1_array)
        plt.title('Optimal state1')
        plt.ylabel('X1 variable')
        plt.xlabel('T variable')
        plt.grid()

        plt.show()

















