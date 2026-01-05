# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name>
<Class>
<Date>
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    singular_values = la.svdvals(A)
    return singular_values[0] / singular_values[-1] if singular_values[-1] != 0 else np.inf # Equation 3 but with a catch for 0


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())

    absolutes = []
    relatives = []

    for _ in range(100):
        rs = [np.random.normal(1, 1e-10) for _ in range(21)]
        new_coeffs = [w_coeffs[i] * rs[i] for i in range(21)] # From step 2
        new_roots = np.roots(new_coeffs) # Feels right
        plt.scatter(np.real(new_roots), np.imag(new_roots), color='black', marker=',', s=0.8) # I don't know why s=0.8 works but it works
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(rs, np.inf)
        absolutes.append(k) # From code block in workbook
        relatives.append(k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)) # Also from workbook

    plt.scatter(range(21), np.zeros_like(range(21)), color='blue', label='Original') # OG
    plt.title("Roots of Perturbed Wilkinson Polynomials")
    plt.xlabel("Real axis")
    plt.ylabel("Imaginary axis")
    plt.legend()
    plt.savefig("prob2.png")

    return np.mean(absolutes), np.mean(relatives) # Let's go


# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1, 1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0], :] = np.inf
        dists[:, index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # From workbook
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    OG_eigs = la.eigvals(A) # Nothing too crazy to see here
    new_eigs = reorder_eigvals(OG_eigs, la.eigvals(A + H)) # From the hint
    absolute = la.norm(OG_eigs - new_eigs, 2) / la.norm(H, 2) # From equation 4

    return absolute, absolute * la.norm(A, 2) / la.norm(OG_eigs, 2) # Also equation 4


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    xs = np.linspace(domain[0], domain[1], res)
    ys = np.linspace(domain[2], domain[3], res)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X) # This is the output

    for i in range(res):
        for j in range(res):
            A = np.array([[1, X[i, j]],[Y[i, j], 1]]) # Make the array
            Z[i, j] = eig_cond(A)[1] # Get the relative condition number

    plt.pcolormesh(X, Y, Z, cmap='gray_r', shading='auto')
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rel. Condition Number of [[1, x],[y, 1]]")
    plt.savefig("prob4.png")


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # From the workbook
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n + 1)

    la_inv_method = la.inv(A.T @ A) @ A.T @ yk # From the workbook

    Q, R = la.qr(A, mode='economic')
    qr_method = la.solve_triangular(R, Q.T @ yk)

    domain = np.linspace(0, 1, 1000)
    plt.scatter(xk, yk, color='green')
    plt.plot(domain, np.polyval(la_inv_method, domain), label="la.inv", color='red')
    plt.plot(domain, np.polyval(qr_method, domain), label="qr")
    plt.legend()
    plt.savefig("prob5.png")

    return la.norm((A @ la_inv_method) - yk, 2), la.norm((A @ qr_method) - yk, 2) # The forward errors


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols('x')

    I_of_n_sym = []
    I_of_n_fact = []

    error = []

    for i in range(1, 11):
        n = int(5 * i)

        f = x**n * sy.exp(x - 1) # To be integrated
        solution = sy.integrate(f, (x, 0, 1))
        I_of_n_sym.append(float(solution))
        fact_solution = (-1 ** n) * (sy.subfactorial(n) - (sy.factorial(n)) / (np.e))
        I_of_n_fact.append(float(fact_solution))

        error.append(float(np.abs(solution - fact_solution) / solution))
    plt.plot([5 * i for i in range(1, 11)], error)
    plt.yscale("log")
    plt.xlabel("n")
    plt.xlabel("error (log)")
    plt.savefig("prob6.png")
