from Tkinter import *
from VariationalProblem import VariationalProblem
from dolfin import *
from newton import *

import sys

N = 0
objectif = 0
constraint = None
penalty = True
target = 0

text_subwindow = None

def solve():
  text_subwindow.delete('1.0',END)
  mesh = UnitIntervalMesh(int(N.get()))
  #penalty = True if penality.get() == 1 else False
  h = None
  if constraint.get() != "":
    h = constraint.get()
    constrained_problem=VariationalProblem(mesh,1,objectif.get().h,Expression(target.get(),degree=1),penalty)
    solver = ipopt.problem(n=len(constrained_problem.x0),m=1,problem_obj=IpoptProblem(constrained_problem),lb=lb,ub=ub,cl=cl,cu=cu)
    solver.addOption(b"mu_strategy", b"adaptive")
    solver.addOption(b"tol", 1e-10)
    solver.addOption(b"max_iter", 1000)
  else:
    unconstrained_problem=VariationalProblem(mesh, 1, objectif.get(),None,Expression(target.get(),degree=1),penalty)
    newton_optim(unconstrained_problem)
    plot(droite.u,interactive=True)


class redirect_stdout(object):
  def __init__(self):
    self.text_subwindow = text_subwindow
  def write(self,string):     
    self.text_subwindow.insert(INSERT,string)
  def flush(self):
    pass

if __name__ == '__main__':
  root = Tk()
  frame_1 = Frame(root)
  L = Label(frame_1,text="Objectif funtion to minimize",anchor="w",width=30)
  L.pack(side=LEFT)
  objectif = Entry(frame_1,width=60)
  objectif.pack(side=LEFT)
  frame_1.pack(anchor="w")

  frame_2 = Frame(root)
  L = Label(frame_2,text="Number of points",anchor="w",width=30)
  L.pack(side=LEFT)
  N = Entry(frame_2,width=60)
  N.pack(side=LEFT)
  frame_2.pack(anchor="w")

  frame_3 = Frame(root)
  L = Label(frame_3,text="Constraint function",anchor="w",width=30)
  L.pack(side=LEFT)
  constraint = Entry(frame_3,width=60)
  constraint.pack(side=LEFT)
  frame_3.pack(anchor="w")

  frame_7 = Frame(root)
  L = Label(frame_7,text="Constraint lower and upper bound",anchor="w",width=30)
  L.pack(side=LEFT)
  wow = Entry(frame_7,width=30)
  wow.pack(side=LEFT)
  wow1 = Entry(frame_7,width=30)
  wow1.pack(side=LEFT)
  frame_7.pack(anchor="w")
  frame_7.pack(anchor="w")

  frame_4 = Frame(root)
  L = Label(frame_4,text="Boundary conditions function",anchor="w",width=30)
  L.pack(side=LEFT)
  target = Entry(frame_4,width=60)
  target.pack(side=LEFT)
  frame_4.pack(anchor="w")

  frame_5 = Frame(root)
  solve_button = Button(frame_5,text="Solve",command=solve)
  solve_button.pack()
  frame_5.pack()

  frame_6 = Frame(root)
  text_subwindow = Text(frame_6)#,state=DISABLED)
  text_subwindow.pack()
  frame_6.pack()



  # hijack sys.stdout.write
  sys.stdout = redirect_stdout()

  root.mainloop()
