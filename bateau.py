from VariationalProblem import VariationalProblem
from IpoptProblem import IpoptProblem
from meshmesh import customMesh
from dolfin import UnitIntervalMesh, DOLFIN_EPS, Constant, inner, grad, Expression, dx, plot
import ipopt
from numpy import *
from newton import newton, newton_optim, armijo

# ---------- The crossing river boat -----------
# Define the function to be integrated
def f(u) : 
    v = Expression("-x[0]*x[0] + x[0]",degree=1)
    c = 1.0 # boat velocity, considered to be constant
    f = ( ( (c**2.0 * (1.0 + inner(grad(u),grad(u))) - v**2.0)**(0.5) - v*inner(grad(u),grad(u))**0.5 ) / ( c**0.5 - v**0.5) )*dx
    return f

# Define Dirichlet boundary (u(x) = 0 @ x = 0)
def boundary(x): # returns true where BC is willed to be applied.
    return x[0] < DOLFIN_EPS 

boundary_val = Constant(1.0)

x_i = 0.0 # starting point
x_f = 1.0 # end point
N = 10 # number of sub-intervals
sensitivePoints = [0.0, 1.0]
nR = 2
mesh = customMesh(x_i, x_f, N, sensitivePoints, nR)
N = mesh.num_cells()
#mesh = UnitIntervalMesh(N)
# 

boat = VariationalProblem(mesh, 1, f, None, boundary, boundary_val)


boat.x0 = ones((N+1), dtype=float_)#arange(2, (N+1)+2, dtype=float_)
for ii in range(1,len(boat.x0),2) :
    boat.x0[ii] = 2.0
#boat.x0[1] = 2.0; boat.x0[3]=2.0; boat.x0[5]=2.0; boat.x0[7]=2.0; boat.x0[9]=2.0; boat.x0[11]=2.0; boat.x0[13]=2.0; boat.x0[15]=2.0; boat.x0[17]=2.0
print boat.x0
print boat.grad(boat.x0)

#quit()

x0 = boat.x0

nvertices = N+1
lb = None #ones((nvertices), dtype=float_)* (0.0)
ub = None #ones((nvertices), dtype=float_)* (10.0)

cl = []
cu = []

solve_boat = ipopt.problem(
            n=len(x0),
            m=len(cl),
            problem_obj=IpoptProblem(boat),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

solve_boat.addOption(b"mu_strategy", b"adaptive")
solve_boat.addOption(b"max_iter", 10000)
solve_boat.addOption(b"tol", 1e-3)

x, info = solve_boat.solve(x0)

print("Solution of the primal variables: x=%s\n" % repr(x))

print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

print("Objective=%s\n" % repr(info['obj_val']))


plot(boat.u, interactive=True)
