# profiling.py
"""Python Essentials: Profiling.
<Name>
<Class>
<Date>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import time
import numpy as np
from math import sqrt
from numba import jit
from matplotlib import pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.


def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    r = len(data) - 2
    while r >= 0: # Base case
        for c in range(r+1): # Have to do one more column than row
            data[r][c] += max(data[r+1][c], data[r+1][c+1]) # Just add rows below
        r -= 1 # Move up a row
    return data[0][0]


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2] # Found the first prime for it I guess lol
    current = 3
    isprime = True
    while len(primes_list) < N:
        for p in primes_list:
            if p**2 > current: # We'll just be done before checking all the primes
                break
            if current % p == 0: # Then it's not prime
                isprime = False
                break
        if isprime:
            primes_list.append(current) # Add it
        current += 2 # Check next odd
        isprime = True # Reset loop
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:, j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm(A - x[:, np.newaxis], axis=0))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    alphabet = {chr(i): i - 64 for i in range(65, 91)} # A fun way to write out the dictionary super quick
    return sum([sum([alphabet[letter] for letter in names[i]]) * (i + 1) for i in range(len(names))])


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    first = 1 # Not quite sure why this works fun fact
    second = 0
    while True:
        first, second = second, first + second # Super speedy quick 🏃💨
        yield second

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    limit = 10**(N-1)
    fibbed = fibonacci()
    while True:
        new = next(fibbed)
        if new >= limit:
            return new


# Problem 6
def prime_sieve(N):
    """Yield all primes less than N."""
    my_list = np.arange(2, N)  # start from 2
    mask = np.ones(len(my_list), dtype=bool) # Make a np array of True values

    for i in range(len(my_list)):
        if mask[i]: # If the spot is a prime
            p = my_list[i]
            yield p
            # eliminate multiples of p
            mask[my_list % p == 0] = False
            mask[i] = True  # keep p itself marked as True


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i, k] * A[k, j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit(nopython=True)
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i, k] * A[k, j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    default_times = []
    jit_times = []
    np_times = []
    for i in range(2, 8):
        m = 2**i
        A = np.random.random((m, m))

        start_default = time.time()
        matrix_power(A, n)
        default_times.append(time.time() - start_default)

        start_jit = time.time()
        matrix_power_numba(A, n)
        jit_times.append(time.time() - start_jit)

        start_np = time.time()
        np.linalg.matrix_power(A, n)
        np_times.append(time.time() - start_np)

    plt.plot(range(2, 8), default_times, color="blue", label="default")
    plt.plot(range(2, 8), jit_times, color="red", label="jit")
    plt.plot(range(2, 8), np_times, color="green", label="np")
    plt.loglog()
    plt.legend()
    plt.title("Run times of various matrix power algorithms")
    plt.savefig("prob7.png")