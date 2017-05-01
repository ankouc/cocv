from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *
import moola
set_log_level(ERROR)
import numpy as np
import matplotlib.pyplot as plt


def boundary_U(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 1.0, tol)

def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], 0.0, tol)


class Optimal_command(object):

    def __init__(self, state_func, solver, bcs, objective, f0, n_vertice=20):
        self.bc_u1 = bcs['bc_u1']
        self.bc_f = bcs['bc_f']
        self.f0 = f0
        self.state_func1 = state_func
        self.n = 50
        self.objective = objective
        self.solver = solver


    def make_function_space(self):
        mesh_state = UnitIntervalMesh(self.n)
        self.mesh_state = mesh_state
        self.W = FunctionSpace(mesh_state, "DG", 0)
        self.V1 = FunctionSpace(mesh_state, "CG", 1)


        self.f = interpolate(Expression(self.f0, degree=1), self.W, name='Control')
        self.u1 = Function(self.V1, name='State1')
        self.v1 = TestFunction(self.V1)
        self.w = TestFunction(self.W)


    def make_boundaries(self, space):
        
        self.bc1 = []
        self.bcf = []


        if self.bc_u1['lower'] is not None:
            bc1_l = DirichletBC(space[0], self.bc_u1['lower'], boundary_L)         
            self.bc1.append(bc1_l)

        if self.bc_u1['upper'] is not None:
            bc1_u = DirichletBC(space[0], self.bc_u1['upper'], boundary_U)
            self.bc1.append(bc1_u)

        if self.bc_f['lower'] is not None:
            bcf_l = DirichletBC(space[1], self.bc_f['lower'], boundary_L)         
            self.bcf.append(bcf_l)

        if self.bc_f['upper'] is not None:
            bcf_u = DirichletBC(space[1], self.bc_f['upper'], boundary_U)
            self.bcf.append(bcf_u)
 


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
                                                           'maxiter': 30})
        else:
            raise NameError('No other solver implemented yet!') 


        solution = solver.solve()
        return solution 

    def condition_string(self):
        print(self.bc1)

        if self.bc1 is not None:
            self.str_cond1 = 'self.bc1'
        else:
            self.str_cond1 = ''

        if self.bcf is not None:
            self.str_condf = 'self.bcf'
        else:
            self.str_condf = ''


    def declare_state_func(self, m):

        if m == 1:
            F_1 = eval(self.state_func1)*dx
        else:
            F_1 = eval(self.state_func1.replace('self.f', 'self.f_opt'))*dx
            
        solve(F_1 == 0, self.u1, eval(self.str_cond1))


        if m == 1 and self.str_condf is not '':
            solve((self.w-self.w)*dx == 0, self.f, eval(self.str_condf))



    def main(self):

	
        self.make_function_space()

        self.make_boundaries([self.V1, self.W])

        self.condition_string()

        self.declare_state_func(1)

        x = SpatialCoordinate(self.mesh_state)

        J = Functional(eval(self.objective))
        control = Control(self.f)

        rf = ReducedFunctional(J, control)
        problem = MoolaOptimizationProblem(rf)
        
        solution = self.solve_pb(problem)

        self.f_opt = solution['control'].data

        self.declare_state_func(2)

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

















