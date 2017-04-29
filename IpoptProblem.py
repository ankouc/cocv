import numpy as np
import ipopt
import scipy.sparse as sps

class IpoptProblem(object):
    def __init__(self, nlp):
        self.nlp = nlp

    def objective(self, x):
        return self.nlp.obj(x)

    def gradient(self, x):
        return self.nlp.grad(x)

    def constraints(self, x):
        return self.nlp.cons(x)

    def jacobian(self, x):
        return self.nlp.jacob(x)

    def hessian(self, x, lagrange, obj_factor):
        return self.nlp.hess(x)

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))
