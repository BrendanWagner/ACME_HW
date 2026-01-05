# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import sparse
from scipy import linalg as la
from matplotlib import pyplot as plt

# Helper function
def diag_dom(n, num_entries=None, as_sparse=False):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.
        as_sparse: If True, an equivalent sparse CSR matrix is returned.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    
    A = sparse.lil_matrix((n,n))
    rows = np.random.choice(n, size=num_entries)
    cols = np.random.choice(n, size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)

    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    B = A.tocsr()               # convert to row format for the next step
    for i in range(n):
        A[i, i] = abs(B[i]).sum() + 1

    return A.tocsr() if as_sparse else A.toarray()

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    D = np.diag(A) # Diagonals as a vector
    x = np.zeros_like(A[0]) # Our initial guess will just be zeros I guess
    tries = [0]
    errors = [1]

    for i in range(maxiter):
        new_x = x + ((b - A @ x) / D) # From equation 2
        errors.append(la.norm(new_x - x, np.inf)) # Add error every time
        tries.append(i + 1) # Add the iteration every time
        if la.norm(new_x - x, np.inf) < tol: # If error is small enough
            break
        x = new_x
    if plot is True: # Just run this if we need a plot
        plt.semilogy(tries, errors)
        plt.savefig("idk.png") # My favorite filename
    return new_x

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    x = np.zeros_like(b) # Initial guess, I guess
    errors = [1]
    tries = [0]
    for i in range(maxiter):
        new_x = x.copy() # Make a copy to check for iteration error
        for j in range(len(x)):
            new_x[j] = new_x[j] + ((b[j] - A[j] @ new_x) / A[j, j]) # From equation 4 maybe?
        errors.append(la.norm(new_x - x, np.inf)) # Add the error
        tries.append(i + 1)
        if la.norm(new_x - x, np.inf) < tol: # If we're done
            break
        x = new_x
    if plot:
        plt.semilogy(tries, errors)
        plt.savefig("idk.png") # You betcha I did idk for this
    return new_x


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_array): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    x = np.zeros_like(b) # Initial guess, I guess
    for i in range(maxiter):
        new_x = x.copy() # Make a copy to check for iteration error
        for j in range(len(x)):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]
            new_x[j] = new_x[j] + ((b[j] - Aix) / A[j, j]) # From equation 4 maybe?
        if la.norm(new_x - x, np.inf) < tol: # If we're done
            print(i)
            break
        x = new_x
    return new_x


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_array): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    x = np.zeros_like(b) # Initial guess, I guess
    converged = False
    after = 0
    for i in range(maxiter):
        new_x = x.copy() # Make a copy to check for iteration error
        for j in range(len(x)):
            rowstart = A.indptr[j]
            rowend = A.indptr[j+1]
            Aix = A.data[rowstart:rowend] @ x[A.indices[rowstart:rowend]]
            new_x[j] = new_x[j] + ((b[j] - Aix) / A[j, j] * omega) # From equation 5 (this is the only change from the last one)
        if la.norm(new_x - x, np.inf) < tol: # If we're done
            converged = True
            after = i + 1
            break
        x = new_x
    return new_x, converged, after


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    I = np.eye(n)
    B = sparse.diags([np.ones(n-1), np.ones(n) * -4, np.ones(n-1)], [-1, 0, 1]) # From workbook
    b = np.zeros(n**2) # Start with just zeroes
    for i in range(n): # Construct b the correct way
        b[i * n] = -100
        b[(i * n) + 3] = -100
    grid = [[None] * n for _ in range(n)] # We'll chuck this into the bmat at the end
    for i in range(n):
        grid[i][i] = B # Diagonals
        if i > 0:
            grid[i][i-1] = I # Up on the diagonal
        if i < n - 1:
            grid[i][i+1] = I # Down on the diagonal
    A = sparse.bmat(grid, format="csr")
    
    u, converged, after = sor(A, b, omega, tol, maxiter)
    if plot:
        u_grid = u.reshape((n, n))
        plt.pcolormesh(u_grid, cmap="coolwarm")
        plt.colorbar()
        plt.savefig("idk.png")
    # print(np.allclose(A @ u, b)) # For testing purposes
    return u, converged, after


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    xs = np.linspace(1, 1.95, 20) # Values of omega
    ys = [] # We'll plot these, but these are iteration counts
    for x in xs:
        _, _, y = hot_plate(20, x, tol=1e-2, maxiter=1000) # We really only care about the iteration count because they all should have the same answer
        ys.append(y)
    plt.plot(xs, ys)