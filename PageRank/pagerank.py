# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import numpy as np
import networkx as nx
from scipy import linalg as la
# from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        new_A = A.copy().astype(float)
        for i in range(len(A)):
            sum = np.sum(A[:, i]) # Get column sum
            if sum:
                new_A[:, i] /= sum
            else:
                new_A[:, i] = 1 / len(A)
        self.A = new_A
        self.n = len(A) # Not necessary but nice
        if labels and len(labels) != self.n: # Check label length
            raise ValueError("Bad labels")
        elif labels is None:
            self.labels = [i for i in range(self.n)]
        else:
            self.labels = labels

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        A = np.eye(self.n) - (epsilon * self.A) # Make A
        b = np.ones(self.n) * (1 - epsilon) / self.n # Make b
        solution = la.solve(A, b) # Solve it
        return {f"{self.labels[i]}": solution[i] for i in range(self.n)}

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        E = np.ones((self.n, self.n))
        B = (epsilon * self.A) + (E * (1 - epsilon) / self.n) # From the workbook
        eig = la.eig(B)[1][:, 0]
        idk = np.sum(eig)
        eig /= idk # Get unit vector
        solution = eig # I don't know why I did this, it's been too long
        return {f"{self.labels[i]}": solution[i] for i in range(self.n)} # Parse it out

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        initial_p = np.ones(self.n) / self.n
        for _ in range(maxiter):
            next_p = ((epsilon * self.A) @ initial_p) + (np.ones(self.n) * (1 - epsilon) / self.n)
            if la.norm(initial_p - next_p, 1) <= tol:
                return {f"{self.labels[i]}": next_p[i] for i in range(self.n)}
            initial_p = next_p
        return {f"{self.labels[i]}": next_p[i] for i in range(self.n)} # Love you Xavier


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    return sorted(d, key=d.get)[::-1] # I don't know why this has to be its own def



# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    with open(filename) as data:
        new_data = []
        labels = []

        for line in data: # Make my labels list
            idk = line.strip().split("/")
            new_data.append(idk)
            for i in idk:
                labels.append(i)
        
        labels = sorted(list(set(labels))) # Sort my labels

        n = len(labels)
        indexed_labels = {labels[i]: i for i in range(n)}
        A = np.zeros((n, n))

        for i in range(len(new_data)): # Looping through data to update adjacency matrix
            line = new_data[i]
            label = line[0]
            for j in line:
                if j == label: # If it's the first one the skip it
                    continue
                else:
                    A[indexed_labels[j], indexed_labels[label]] = 1 # why the fetch is it this indexing

        idk = DiGraph(A, labels)
        return get_ranks(idk.itersolve(epsilon=epsilon))
            


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    with open(filename) as data:
        new_data = []
        teams = set()
        next(data)
        for line in data:
            winner, loser = line.strip().split(",")
            new_data.append([winner, loser])
            teams.add(winner) # Adding to the set
            teams.add(loser)
        
        teams = list(teams) # Make it a list so I can iterate
        n = len(teams)
        indexed_teams = {teams[i]: i for i in range(n)} # Like in the hint
        A = np.zeros((n, n))

        for line in new_data:
            winner, loser = line
            A[indexed_teams[winner], indexed_teams[loser]] += 1 # Give a dub to the team

        idk = DiGraph(A, teams) # Same as last time
        return get_ranks(idk.itersolve(epsilon=epsilon))



# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    with open(filename) as data:
        idk = nx.DiGraph() # Gotta love the variable names
        for line in data: # Iterate through each movie
            _, *actors = line.strip().split("/") # Get rid of the title then keep the characters. Weird syntax but here we are
            for i in range(len(actors)): # Loop through actors 
                for j in range(i + 1, len(actors)): # Loop through REST of actors
                    if idk.has_edge(actors[j], actors[i]):
                        idk[actors[j]][actors[i]]["weight"] += 1 # Update weight
                    else:
                        idk.add_edge(actors[j], actors[i], weight = 1) # Make new edge
    return get_ranks(nx.pagerank(idk, alpha=epsilon)) # Love you Xavier
