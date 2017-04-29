# -*- coding: utf-8 -*-
"""A very simple Newton method for unconstrained optimization."""

import numpy as np
from dolfin import plot


def newton(nlp):
    """Apply Newton's method to a `NLPModel`.

    :parameters:
        :nlp: an `NPLModel` instance.
    """
    x = nlp.x0
    f = nlp.obj(x)
    g = nlp.grad(x)
    gNorm = np.linalg.norm(g)
    gNorm0 = gNorm
    k = 0

    hdr = ("it", "f", u"‖∇f‖", u"‖Δx‖", u"cos(Θ)", "t")
    print "%2s  %9s  %8s  %8s  %8s  %7s" % hdr
    print "%2d  %9.2e  %8.2e" % (k, f, gNorm),

    while k < 15 and gNorm > 1.0e-8 + 1.0e-6 * gNorm0:
        H = nlp.hess(x)

        # Compute the Newton step.
        dx = np.linalg.solve(H, g)
        slope = np.dot(dx, g)
        assert(slope > 0)

        # Perform Armijo linesearch.
        t = 1.0
        xp = x - t * dx
        fp = nlp.obj(xp)
        while fp > f - 1.0e-4 * t * slope and t > 1.0e-6:
            t = t / 1.5
            xp = x - t * dx
            fp = nlp.obj(xp)

        x = xp
        dxNorm = np.linalg.norm(dx)
        print "  %8.2e  %8.1e  %7.1e" % (dxNorm, -slope / dxNorm / gNorm, t)

        f = fp
        g = nlp.grad(xp)
        gNorm = np.linalg.norm(g)

        k = k + 1
        print "%2d  %9.2e  %8.2e" % (k, f, gNorm),
    print
    return


def armijo(xk, dk, nlp):
    fk = nlp.obj(xk)
    gk = nlp.grad(xk)
    slope = np.dot(gk, dk)  # Doit être < 0
    t = 1.0
    while nlp.obj(xk + t * dk) > fk + 1.0e-4 * t * slope:
        t /= 1.5
    return t

def newton_optim(nlp):
    xk = nlp.x0
    n  = nlp.x0.size
    fk = nlp.obj(xk)
    gk = nlp.grad(xk)
    gNorm = np.linalg.norm(gk)
    gNorm0 = gNorm
    Hk = nlp.hess(xk)
    k = 0
    print "%2d  %9.2e  %7.1e" % (k, fk, gNorm)
    while gNorm > 1.0e-6 * gNorm0 and k < 20:
        dk = -np.linalg.solve(Hk, gk)
        slope = np.dot(gk, dk)
        mult = 0.0
        while slope >= 1.0e-4 * np.linalg.norm(dk) * gNorm:
            dk = -np.linalg.solve(Hk + mult * np.eye(n), gk)
            slope = np.dot(gk, dk)
            mult = max(1.0e-3, mult * 10)
        t = armijo(xk, dk, nlp)
        xk += t * dk
        fk = nlp.obj(xk)
        gk = nlp.grad(xk)
        gNorm = np.linalg.norm(gk)
        Hk = nlp.hess(xk)
        k += 1
        print "%2d  %9.2e  %7.1e  %7.1e  %7.1e" % (k, fk, gNorm, t, mult)
    return xk

