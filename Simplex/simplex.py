"""Volume 2: Simplex

<Name>
<Date>
<Class>
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        x = np.zeros_like(c) # The origin
        if np.min(b) < 0: # Check feasibility
            raise ValueError("Not feasible at the origin") # Thanks Chammelia
        self.c = c
        self.A = A
        self.b = b
        self.n = len(c) # Not mandatory but I gotta say it's handy having these here
        self.m = len(b)
        self._generatedictionary(self.c, self.A, self.b) # Make the dictionary

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        A_hat = np.hstack((A, np.eye(self.m)))
        first_row = np.hstack((np.array([0]), c, np.zeros(self.m)))
        second_row = np.hstack((b.reshape(-1, 1), -1 * A_hat)) # I still don't understand this syntax but here we are
        self.dictionary = np.vstack((first_row, second_row))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        row = self.dictionary[0, 1:]  # Skip the zero
        if np.all(row >= 0):
            return None  # Hopefully we never need this
        return int(np.argmin(row)) + 1  # + 1 because we skipped the first zero

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        first_column = -1 * self.dictionary[:, 0]
        j_th_column = self.dictionary[:, index]
        ratios = [np.inf if j_th_column[i] >= 0 or i == 0 else first_column[i] / j_th_column[i] for i in range(len(first_column))] # This is gross but it basically just takes all of the ratio tests except it cuts to np.inf anytime the condiitons aren't met. This makes it easy to find the index that we actually want.
        return np.argmin(ratios)

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        col = self._pivot_col()
        row = self._pivot_row(col)
        assert np.any(self.dictionary[:, col] <= 0), ValueError("Unbounded, silly")
        pivot_val = self.dictionary[row, col]
        self.dictionary[row, :] /= -pivot_val
        for i in range(len(self.dictionary)): # I'm sure I could do this using array broadcasting but I don't have the bandwidth today
            if i != row: # Skip own row
                self.dictionary[i, :] -= self.dictionary[row, :] * self.dictionary[i, col] / self.dictionary[row, col]

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        while np.any(self.dictionary[0, 1:] < 0): # Just solve it
            self.pivot()
            
        A = self.dictionary[1:, 1:]
        b = self.dictionary[1:, 0]
        basic_vars = {}
        nonbasic_vars = {}
        for j in range(A.shape[1]):
            col = A[:, j]
            if np.count_nonzero(np.isclose(col, 0)) == (len(col) - 1) and np.isclose(abs(col).max(), 1):
                i = np.argmax(abs(col))
                basic_vars[j] = b[i]
            else:
                nonbasic_vars[j] = 0.0
        obj_value = self.dictionary[0, 0]
        return obj_value, basic_vars, nonbasic_vars

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    data = np.load(filename)

    p = data["p"]
    m = data["m"]
    d = data["d"]

    top_A = data["A"]
    bottom_A = np.eye(len(p))

    A = np.vstack((top_A, bottom_A)) # This is so I can make b be the material constraint AND the demand

    c = -1 * p # idk brochacho
    b = np.hstack((m, d))

    idk = SimplexSolver(c, A, b)
    wow = idk.solve()
    new = wow[1] # Just get the dictionary

    return [new[0], new[1], new[2], new[3]] # Just the first four of the dictionary