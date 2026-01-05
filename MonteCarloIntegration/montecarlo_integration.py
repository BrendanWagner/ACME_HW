# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import scipy


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    vol = 2**n
    points = np.random.uniform(-1, 1, (n, N))
    lengths = scipy.linalg.norm(points, axis=0)
    num_within = np.count_nonzero(lengths < 1) # Check if it's in the ball
    return vol * num_within / N

# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    vol = b - a # Feels intuitive
    points = np.random.uniform(a, b, (N)) # Thank goodness this notation makes sense
    heights = np.sum(f(points))
    return vol * heights / N


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    n = len(mins)
    vol = np.prod([maxs[i] - mins[i] for i in range(n)])
    points = np.array(np.random.uniform(0, 1, (n, N))) # n rows of N points
    for i in range(n):
        points[i] *= maxs[i] - mins[i] # Stretch
        points[i] += mins[i] # Shift
    points = points.T # This feels easier
    mins = np.array(mins)
    mins = np.array(maxs)
    evaluated_points = [f(points[j]) for j in range(N)]
    return vol * sum(evaluated_points) / N


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    f = lambda x: (1 / ((2 * np.pi)**(len(x) / 2))) * np.exp((x.T @ x) / (-2)) # Hope this works. I don't know how to test it though lol
    mins = [-1.5, 0, 0, 0]
    maxs = [0.75, 1, 0.5, 1]
    mean, cov = np.zeros(4), np.eye(4)
    actual = stats.multivariate_normal.cdf(maxs, lower_limit=mins, mean=mean, cov=cov) # From the workbook
    guess_counts = np.logspace(1, 5, 20) # Not integers but I'll correct for that in a minute
    errors = [np.abs(actual - mc_integrate(f, np.array(mins), np.array(maxs), int(j))) / np.abs(actual) for j in guess_counts] # See I corrected it

    plt.plot(guess_counts, errors, label='Relative Error (Monte Carlo)', linestyle='-', marker='o')
    plt.plot(guess_counts, 1 / np.sqrt(guess_counts), label='1/sqrt(N)', linestyle='-', marker='o')
    plt.title("Monte Carlo Convergence")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("prob4.png")

prob4()