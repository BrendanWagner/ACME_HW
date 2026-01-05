# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""

import time
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt

from jax import numpy as jnp
from jax import grad
import jax


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols("x")
    expr = (sy.sin(x) + 1)**(sy.sin(sy.cos(x))) # From docstring
    diff = sy.diff(expr, x) # Take derivative
    # expr_lam = sy.lambdify(x, expr)
    diff_lam = sy.lambdify(x, diff) # Lambdify
    return diff_lam


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x - h)) / h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x + h) - f(x - h)) / (2 * h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    domain = np.logspace(-8, 0, 9)
    f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x))) # Apparently I need this again
    f_prime = prob1() # Feels intuitive

    # Calculate errors time
    first_errors = [np.abs(f_prime(x0) - fdq1(f, x0, power)) for power in domain]
    second_errors = [np.abs(f_prime(x0) - fdq2(f, x0, power)) for power in domain]
    third_errors = [np.abs(f_prime(x0) - bdq1(f, x0, power)) for power in domain]
    fourth_errors = [np.abs(f_prime(x0) - bdq2(f, x0, power)) for power in domain]
    fifth_errors = [np.abs(f_prime(x0) - cdq2(f, x0, power)) for power in domain]
    sixth_errors = [np.abs(f_prime(x0) - cdq4(f, x0, power)) for power in domain]

    # Plotting time
    plt.loglog(domain, first_errors, "-o", label="Order 1 Forward", color="blue")
    plt.loglog(domain, second_errors, "-o", label="Order 2 Forward", color="orange")
    plt.loglog(domain, third_errors, "-o", label="Order 1 Backward", color="green")
    plt.loglog(domain, fourth_errors, "-o", label="Order 2 Backward", color="red")
    plt.loglog(domain, fifth_errors, "-o", label="Ordr 2 Centered", color="purple")
    plt.loglog(domain, sixth_errors, "-o", label="Order 4 Centered", color="brown")
    plt.legend()
    plt.title("h-size vs. Error")
    plt.savefig("prob3.png")


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    data = np.load("plane.npy") # Get the data
    data[:, 1:] = np.deg2rad(data[:, 1:]) # Convert to radians
    xs = [500 * (np.tan(beta)) / (np.tan(beta) - np.tan(alpha)) for alpha, beta in data[:, 1:]] # From equation 4
    ys = [500 * (np.tan(beta) * np.tan(alpha)) / (np.tan(beta) - np.tan(alpha)) for alpha, beta in data[:, 1:]] # From equation 4
    first_x_prime = xs[1] - xs[0] # Forward speed
    last_x_prime = xs[7] - xs[6] # Backward speed
    rest_of_x_primes = [(xs[i+1] - xs[i-1]) / (2) for i in range(len(xs))[1:-1]] # Use the centered thing
    first_y_prime = ys[1] - ys[0] # Forward again
    last_y_prime = ys[7] - ys[6] # Backward again
    rest_of_y_primes = [(ys[i+1] - ys[i-1]) / (2) for i in range(len(ys))[1:-1]] # Centered thing again
    x_prime = [first_x_prime] + rest_of_x_primes + [last_x_prime] # Stitch all together
    y_prime = [first_y_prime] + rest_of_y_primes + [last_y_prime]
    velocities = [np.sqrt(x**2 + y**2) for x, y in zip(x_prime, y_prime)] # Calculate velocities
    return np.array(velocities)


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n = x.size
    m = f(x).size
    jacobian = np.zeros((m, n))
    I = np.identity(n)
    for i in range(n):
        new_result = (f(x + (h * I[:, i])) - f(x - (h * I[:, i]))) / (2 * h) # Get the function evaluated in the i direction
        for j in range(m): # I think I flipped i and j but it'll work out
            jacobian[j, i] = new_result[j] # Set the entry of the Jacobian
    return jacobian


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
        return jnp.ones_like(x) # Covering our bases
    if n == 1:
        return x # Pretty straight forward
    
    T_0 = jnp.ones_like(x)
    T_1 = x

    for _ in range(2, n + 1): # I hate the notation here but pretty much do 2 less than n loops
        T_2 = 2 * x * T_1 - T_0 # Some wicked fast computation
        T_0, T_1 = T_1, T_2 # Reassign for next loop

    return T_1

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    x = jnp.linspace(-1, 1, 200)  # points to evaluate

    plt.figure(figsize=(8,6))

    for n in range(5):
        cheb_grad = jax.vmap(grad(lambda x: cheb_poly(x, n))) # Instructions were horrible so I did what worked
        y_prime = cheb_grad(x) # Don't know why this works

        plt.plot(x, y_prime, label=f"T_{n}'(x)")

    plt.title("Derivatives of Chebyshev Polynomials T_n'(x)")
    plt.xlabel("x")
    plt.ylabel("Derivative")
    plt.legend()
    plt.grid(True)
    plt.savefig("prob6.png")


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    f = lambda x: (np.sin(x) + 1)**(np.sin(np.cos(x))) # Apparently I need this again
    jf = lambda x: (jnp.sin(x) + 1)**(jnp.sin(jnp.cos(x))) # Apparently I need this again

    sym_times = []
    sym_accuracies = []

    diff_times = []
    diff_accuracies = []

    jax_times = []
    jax_accuracies = []

    for i in range(N):
        # Step 1
        x_0 = np.random.rand()

        # Step 2
        exact_start = time.time()
        exact_lambda = prob1()
        exact_value = exact_lambda(x_0)
        exact_end = time.time()

        # Step 3
        diff_start = time.time()
        diff_value = cdq4(f, x_0)
        diff_end = time.time()
        diff_error = np.abs(diff_value - exact_value)

        # Step 4
        jax_start = time.time()
        jax_value = grad(jf)(x_0)
        jax_end = time.time()
        jax_error = np.abs(jax_value - exact_value)

        sym_times.append(exact_end - exact_start)
        sym_accuracies.append(0.000000000000000001)

        diff_times.append(diff_end - diff_start)
        diff_accuracies.append(diff_error)

        jax_times.append(jax_end - jax_start)
        jax_accuracies.append(jax_error)

    plt.scatter(sym_times, sym_accuracies, label='Sym')
    plt.scatter(diff_times, diff_accuracies, label='Diff')
    plt.scatter(jax_times, jax_accuracies, label='Jax')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Computation time vs. error")
    plt.xlabel("Computation time (seconds)")
    plt.ylabel("Absolute Error")
    plt.savefig("prob7.png")
    # Looks like my diff equations were way faster than yours??