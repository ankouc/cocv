from VariationalProblem import VariationalProblem
from dolfin import *
import numpy as np
from newton import newton, newton_optim

nvertices=200 #number of vertices
deg=1 #degree of Lagrange polynomial
mesh= UnitIntervalMesh(nvertices-1) 
target=Expression("2.0 - x[0]/4.0",degree=deg)#boundary condition u0
f='(0.5/9.8)**(0.5) * ( (1+inner(grad(u),grad(u)))/(2 - u) )**(0.5)*dx'
h=None #no constraint
penalty=True #add penalty to functional

brachistochrone=VariationalProblem(mesh, deg, f, h,target,penalty)

newton_optim(brachistochrone) #call solver
plot(brachistochrone.u, interactive=True)


