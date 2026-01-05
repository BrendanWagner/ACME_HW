"""Unit testing file for the Simplex lab"""

import simplex
import numpy as np

def test_simplex():
    """
    Tests Simplex using problem 13.19 from the texbook, minimum should be  at x* = (15, 12), minimum value is -132
    """
    c = np.array([-4, -6])
    b = np.array([11, 27, 90])
    A = np.array([[-1, 1], [1, 1], [2, 5]])

    solver = simplex.SimplexSolver(c, A, b)
    sol = solver.solve()
    print(sol)

    # Checks if it returned the correct value
    assert sol[0] == -132, "Incorrect result from the simplex method"


def test_simplex_example():
    # Sets up the values for the simplex problem.
    c = np.array([-3, -2])
    b = np.array([2, 5, 7])
    A = np.array([[1, -1], [3, 1], [4, 3]])

    # Runs the simplex solver.
    solver = simplex.SimplexSolver(c, A, b)
    sol = solver.solve()

    # Checks if it returned the correct value
    assert sol[0] == -5.2, "Incorrect result from the simplex method"
