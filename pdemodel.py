# -*- coding: utf-8 -*-
"""PDE-Constrained Optimization Problems.

A generic class to define a PDE-constrained optimization problem using FEniCS
D. Orban, 2010--2016.
"""

from nlpmodel import NLPModel
import numpy as np
from dolfin import Function, TestFunction, TrialFunction, \
                   Constant, \
                   project, assemble, derivative, action
import logging

__docformat__ = 'restructuredtext'


# Silence FFC.
ffc_logger = logging.getLogger('FFC')
ffc_logger.setLevel(logging.WARNING)


class PDENLPModel(NLPModel):
    """A generic class to represent PDE-constrained optimization problems."""

    def __init__(self,
                 mesh,
                 function_space,
                 bcs,
                 name="Generic PDE model",
                 **kwargs):
        """Initialize a PDE-constrained optimization model.

        :parameters:
            :mesh: a mesh over the domain
            :function_space: a ``FunctionSpace`` instance
            :bcs: a list of boundary conditions

        :keywords:
            :name: a name given to the problems

        An initial guess may be specified via the keyword argument 'u0' as an
        ``Expression`` or ``Constant`` instance. By default,
        ``u0 = Constant(0)``.

        See ``NLPModel`` for other keyword arguments.
        """
        # Obtain initial guess and problem size.
        n = function_space.dim()
        self.__u0 = kwargs.get("u0", Constant(0))
        self.__u = Function(function_space)
        self.__u.assign(self.__u0)

        # Initialize base class.
        super(PDENLPModel, self).__init__(n,
                                          m=0,
                                          name=name,
                                          x0=self.__u.vector().array(),
                                          **kwargs)
        self._sparse_coord = False

        self.__mesh = mesh
        self.__function_space = function_space
        self.__bcs = bcs

        self._objective_functional = None
        self.__objective_gradient = None
        self.__objective_hessian = None

        self._constraint_functional = None
        self.__constraint_jacobian = None

        return

    @property
    def mesh(self):
        """Return underlying mesh."""
        return self.__mesh

    @property
    def function_space(self):
        """Return underlying function space."""
        return self.__function_space

    @property
    def bcs(self):
        """Return problem boundary conditions."""
        return self.__bcs

    @property
    def u0(self):
        """Initial guess."""
        return self.__u0

    @property
    def u(self):
        """Current guess."""
        return self.__u

    def register_objective_functional(self):
        """Register a functional as objective function."""
        raise NotImplementedError("Must be subclassed.")

    def assign_vector(self, x, apply_bcs=True):
        """Assign NumPy array or ``Expression`` ``x`` as current ``u``."""
        if isinstance(x, np.ndarray):
            self.u.vector()[:] = x
        else:
            xx = project(x, self.function_space)
            self.u.vector()[:] = xx.vector().array()

        # Apply boundary conditions if requested.
        if apply_bcs:
            for bc in self.__bcs:
                bc.apply(self.u.vector())

        return

    def obj(self, x, apply_bcs=True):
        """Evaluate the objective functional at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.
        """
        if self._objective_functional is None:
            msg = "Subclass and implement register_objective_functional()."
            raise NotImplementedError(msg)

        self.assign_vector(x, apply_bcs=apply_bcs)
        return assemble(self._objective_functional)

    def compile_objective_gradient(self):
        """Compute first variation of objective functional.

        This method only has side effects.
        """
        if self._objective_functional is None:
            raise NotImplementedError("Objective functional not registered.")

        # Fast return if first variation was already compiled.
        if self.__objective_gradient is not None:
            return

        du = TestFunction(self.function_space)
        self.__objective_gradient = derivative(self._objective_functional,
                                               self.u, du)
        return

    def grad(self, x, apply_bcs=True):
        """Evaluate the gradient of the objective functional at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          x = Expression('x[0] * sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.

        .. todo::

            Why doesn't this work with a Constant() ?
        """
        if self.__objective_gradient is None:
            self.compile_objective_gradient()

        self.assign_vector(x, apply_bcs=apply_bcs)

        # Evaluate gradient and apply boundary conditions.
        g = assemble(self.__objective_gradient)
        # for bc in self.bcs:
        #     bc.apply(g)

        return g.array()

    def compile_objective_hessian(self):
        """Compute second variation of objective functional.

        This method only has side effects.
        """
        # Fast return if second variation was already compiled.
        if self.__objective_hessian is not None:
            return

        # Make sure first variation was compiled.
        if self.__objective_gradient is None:
            self.compile_objective_gradient()

        du = TrialFunction(self.function_space)
        self.__objective_hessian = derivative(self.__objective_gradient,
                                              self.u, du)
        return

    def hess(self, x, y=None, apply_bcs=True, **kwargs):
        """Evaluate the Hessian matrix of the objective functional at ``x``.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          v = Expression('x[0]*sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.

        .. todo::

            Why doesn't this work with a Constant() ?
        """
        obj_weight = kwargs.get('obj_weight', 1.0)

        if self.__objective_hessian is None:
            self.compile_objective_hessian()

        self.assign_vector(x)
        H = assemble(self.__objective_hessian)
        # for bc in self.bcs:
        #     bc.apply(H)

        return obj_weight * H.array()

    def hprod(self, x, y, v, apply_bcs=True, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.

        :parameters:
            :x: NumPy ``array`` or Dolfin ``Expression``
            :y: Numpy ``array`` or Dolfin ``Expression`` for multipliers
            :apply_bcs: apply boundary conditions to ``x`` prior to evaluation.

        If ``x`` is an ``Expression``, it must defined on the appropriate
        function space, by, e.g., declaring it as::

          v = Expression('x[0]*sin(x[1])',
                         element=pdenlp.function_space.ufl_element())

        where ``pdenlp`` is a ``PDENPLModel`` instance.
        """
        obj_weight = kwargs.get('obj_weight', 1.0)

        if self.__objective_hessian is None:
            self.compile_objective_hessian()

        self.assign_vector(x)

        w = Function(self.function_space)
        w.vector()[:] = v
        Hop = action(self.__objective_hessian, w)
        hv = assemble(Hop)
        return obj_weight * hv.array()



# ---------------------------
# Section ajoutée par Hadrien
    def cons(self, x):
        if self._constraint_functional is None:
            msg = "Subclass and implement register_constraint_functional()."
            raise NotImplementedError(msg)

        self.assign_vector(x) # is this really needed? What is its use?
        return assemble(self._constraint_functional)


    def compile_constraint_jacobian(self): # inspired from the 
        """Compute Jacobian of the constraints vector.

        This method only has side effects.
        """
        # Fast return if Jacobian was already compiled.
        if self.__constraint_jacobian is not None:
            return

        # Make sure a constraint was specified.
        if self._constraint_functional is None:
            print("No constraint was specified")

        du = TestFunction(self.function_space)
        dh = TrialFunction(self.function_space)
        self.__constraint_jacobian = derivative(self._constraint_functional, self.u, du)
        return 


    def jacob(self, x, apply_bcs=True) :
        if self.__constraint_jacobian is None:
            self.compile_constraint_jacobian()

        self.assign_vector(x, apply_bcs=apply_bcs)

        # Evaluate gradient and apply boundary conditions.
        j = assemble(self.__constraint_jacobian)
        # for bc in self.bcs:
        #     bc.apply(g)

        return j.array()

