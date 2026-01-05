# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""

import numpy as np
import cvxpy as cp

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    G = np.array([[1, 2, 0],[0, 1, -4]])
    P = np.vstack((np.array([2, 10, 3]), np.eye(3)))
    h = np.array([3, 1])
    q = np.array([12, 0, 0, 0])

    constraints = [G @ x <= h, P @ x >= q] # Don't know why this syntax works but I think I got the right answer

    problem = cp.Problem(objective, constraints)
    optimal_value = problem.solve()
    return x.value, optimal_value



# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]), nonneg = True) # Literally the same thing as the last problem I don't understand
    constraints = [A @ x == b]
    objective = cp.Minimize(cp.norm(x, 1))
    problem = cp.Problem(objective, constraints)
    optimal_value = problem.solve()
    return x.value, optimal_value



# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    movement = cp.Variable((3, 2), nonneg = True, integer = True) # Need the integer
    cost = np.array([[4, 6, 8], [7, 8, 9]]).T # I put it in wrong then was too lazy to transpose it by hand
    supply = np.array([7, 2, 4])
    demand = np.array([5, 8])
    constraints = [movement.sum(axis=1) == supply, movement.sum(axis=0) == demand] # Feels intuitive
    objective = cp.Minimize(cp.sum(cp.multiply(cost, movement)))
    problem = cp.Problem(objective, constraints)
    idk = problem.solve() # Because why would I call it something normal
    return movement.value, idk # Bingo


# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]]) # Taken from formula
    r = np.array([3, 0, 1]) # Easier part to take from formula
    x = cp.Variable(3) # Cuz we need it
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x)) # From the workbook
    idk = prob.solve() # My favorite variable name
    return x.value, idk


# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A[0]), nonneg = True)
    constraints = [x.sum() == 1, x >= 0] # From the workbook I guess
    objective = cp.Minimize(cp.norm(A @ x - b, 2)) # Same as usual
    problem = cp.Problem(objective, constraints) # Literally copied and pasted this
    idk = problem.solve() # Why am I here
    return x.value, idk


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    data = np.load("food.npy", allow_pickle=True).T # I feel like this makes way more sense transposed
    price = data[0, :] # First row. Very important
    servings = data[1, :] # First row. Very important
    calories_fat_sugar = data[2:5, :] * servings # The things we want to cap
    calc_fiber_protein = data[5:, :] * servings # The things we want to exceed
    x = cp.Variable(len(price), nonneg = True, integer = True) # Needed to be an integer I think
    constraints = [calories_fat_sugar @ x <= np.array([2000, 65, 50]), calc_fiber_protein @ x >= np.array([1000, 25, 50])] # From workbook
    objective = cp.Minimize(x @ price) # Ya know
    problem = cp.Problem(objective, constraints) # Once again
    idk = problem.solve() # My favorite variable!
    return x.value, idk