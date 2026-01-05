# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import optimize as opt
from matplotlib import pyplot as plt

# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    for i in range(maxiter):
        new_df = Df(x0) # Get Df applied to current point
        idk = lambda alpha: f(x0 - (alpha * new_df)) # I hate this notation
        alpha = opt.minimize_scalar(idk)["x"] # What is happening
        if la.norm(new_df, np.inf) < tol: # If it converges
            return x0, True, i + 1
        x0 = x0 - (alpha * new_df) # Iterate
    return x0, False, maxiter


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = (Q @ x0) - b
    d0 = -1 * r0
    k = 0
    while la.norm(r0, np.inf) >= tol and k < len(b): # From pseudocode
        alphak = (r0 @ r0) / (d0 @ Q @ d0)
        x0 = x0 + (alphak * (Q @ d0))
        next_r0 = r0 + (alphak * d0)
        beta0 = next_r0 @ next_r0 / r0 @ r0
        d0 = -1 * next_r0 + beta0 * d0
        r0 = next_r0
        k += 1
    if k < len(b): # Check to see if it converged
        return x0, True, k
    else:
        return x0, False, k


# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    r0 = -1 * df(x0)
    d0 = r0
    idk = lambda alpha: f(x0 + (alpha * d0)) # This again, eh?
    alpha = opt.minimize_scalar(idk)["x"] # Yes, this again
    x0 = x0 + alpha * d0
    k = 1
    while la.norm(r0, np.inf) >= tol and k < maxiter:
        new_r0 = -df(x0)
        beta = (new_r0 @ new_r0) / (r0 @ r0)
        d0 = new_r0 + beta * d0
        r0 = new_r0
        alpha = opt.minimize_scalar(idk)["x"]
        x0 = x0 + alpha * d0
        k += 1
    if k < maxiter:
        return x0, True, k
    else: 
        return x0, False, k

