from VariationalProblem import VariationalProblem
from dolfin import *
from newton import newton, newton_optim, armijo
import numpy as np

N=60 #number of vertices
mesh = UnitIntervalMesh(N) 
deg = 1 #degree of Lagrange polynomial
f='(1 + inner(grad(u),grad(u)))**(0.5) * dx' # functional
penalty=True #apply penalty to functional
h=None # no constraints
d=2
target = Expression('x[0]',degree=deg)

droite=VariationalProblem(mesh, deg, f, h,target,penalty) 

newton_optim(droite)
plot(droite.u, interactive=True)
