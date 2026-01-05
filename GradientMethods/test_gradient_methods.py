"""Unit testing file for Gradient Descent Methods lab"""

import gradient_methods
import numpy as np
from scipy import optimize as opt
from scipy import linalg as la

def test_nonlinear_conjugate_gradient():
    """
    Write at least one unit test for problem 3, the nonlinear conjugate gradient function.
    """
    f = lambda x: np.sum([x[i]**4 for i in range(len(x))])
    Df = lambda x: np.array([4 * x[i]**3 for i in range(len(x))])
    x0 = np.array([1, 2, 3])
    my_value = gradient_methods.nonlinear_conjugate_gradient(f, Df, x0, maxiter=10)[0]
    opt_value = opt.fmin_cg(f, x0, fprime=Df)
    assert la.norm(my_value - opt_value, np.inf) < 1e-5, "Incorrect final value"

def test_conjugate_gradient():
    # Tests different sized matrices to see if they converge
    for n in range(1, 5):
        A = np.random.random((n, n))
        b = np.random.random(n)
        Q = A.T @ A
        x, conv, k = gradient_methods.conjugate_gradient(Q, b, np.random.random(n))
        if conv:
            assert np.allclose(Q @ x, b), "Incorrect vector found"