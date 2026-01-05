"""Unit testing file for the PageRank lab"""


import pagerank
import numpy as np

def test_prob2():
    """
    Write unit tests to test the eigensolve and itersolve methods for problem 2.
    """
    my_A = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]])
    my_idk = pagerank.DiGraph(my_A)
    my_sol = {'0': np.float64(0.37012987012987025), '1': np.float64(0.2597402597402597), '2': np.float64(0.3701298701298701)}

    their_A = np.array([[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0]])
    their_labels = ["a", "b", "c", "d"]
    their_idk = pagerank.DiGraph(their_A, labels=their_labels)
    their_sol = {'a': np.float64(0.09575863576738076), 'b': np.float64(0.27415828596414515), 'c': np.float64(0.3559247923043289), 'd': np.float64(0.2741582859641452)}

    assert np.allclose(my_idk.linsolve(), my_sol), "Wrong solution"
    assert np.allclose(my_idk.eigensolve(), my_sol), "Wrong solution"
    assert np.allclose(my_idk.itersolve(), my_sol), "Wrong solution"

    assert np.allclose(their_idk.linsolve(), their_sol), "Wrong solution"
    assert np.allclose(their_idk.eigensolve(), their_sol), "Wrong solution"
    assert np.allclose(their_idk.itersolve(), their_sol), "Wrong solution"

def test_linsolve():
    # Sets up the matrix in the example.
    A = np.array([[0, 0, 0, 0],
                  [1, 0, 1, 0],
                  [1, 0, 0, 1],
                  [1, 0, 1, 0]])
    
    # Sets up the class to be used for the pagerank
    dg1 = pagerank.DiGraph(A, labels=["a", "b", "c", "d"])

    # Finds the p vector.
    p = np.array(list(dg1.linsolve().values()))

    # Checks that the p vector sums to 1 and has the correct values.
    assert np.isclose(p.sum(), 1), "p vector doesn't sum to 1"
    assert np.allclose(p, np.array([0.095758635, 0.274158285, 0.355924792, 0.274158285])), "p vector returns the incorrect values"
