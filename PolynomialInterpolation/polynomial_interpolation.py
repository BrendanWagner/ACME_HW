# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""

import numpy as np
from numpy.fft import fft
from scipy.interpolate import BarycentricInterpolator
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    interp_values = np.zeros(len(points))
    for i in range(len(xint)):
        # Denominator for L_i
        denom = np.prod([xint[i] - xint[j] for j in range(len(xint)) if i != j])
        
        for k in range(len(points)):
            # Numerator for L_i(points[k])
            num = np.prod([points[k] - xint[j] for j in range(len(xint)) if i != j])
            
            # Add contribution from this basis polynomial
            interp_values[k] += (num / denom) * yint[i]

    xint_f = np.linspace(-1, 1, 1000)
    yint_f = 1 / (1 + (25 * (xint_f ** 2)))
    plt.plot(xint_f, yint_f, label="Grunge's")
    plt.plot(points, interp_values, label=f"Interp with {len(xint)} points")
    plt.legend()
    plt.savefig("prob1.png")

    return interp_values


# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.xint = xint # Save xint
        self.yint = yint # Save yint
        self.n = len(xint) # I don't want to calculate this a million times
        self.weights = np.ones(self.n) # Initialize weights
        C = (np.max(self.xint) - np.min(self.xint)) / 4 # Just calculate C once
        for i in range(self.n):
            self.weights[i] = 1 / np.prod([1 / C if i == j else (self.xint[i] - self.xint[j]) / C for j in range(self.n)]) # Array broadcasting for equation 3

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        evaluated_points = np.zeros(len(points)) # Initialize
        for x in range(len(points)):
            if points[x] in self.xint:  # Handle exact interpolation node
                idk = np.where(self.xint == points[x])[0][0] # Not too crazy
                evaluated_points[x] = self.yint[idk]
            else:
                num = 0.0
                denom = 0.0
                for i in range(self.n):
                    num += self.weights[i] * self.yint[i] / (points[x] - self.xint[i]) # Add them all up
                    denom += self.weights[i] / (points[x] - self.xint[i])
                evaluated_points[x] = num / denom

        xint_f = np.linspace(-1, 1, 1000) 
        yint_f = 1 / (1 + (25 * (xint_f ** 2))) # Actual plot
        plt.plot(xint_f, yint_f, label="Grunge's")
        plt.plot(points, evaluated_points, label=f"Interp with {len(points)} points")
        plt.legend()
        plt.savefig("prob3.png")
        return evaluated_points

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        self.xint = np.hstack([self.xint, xint])
        self.yint = np.hstack([self.yint, yint])
        new_weights = np.zeros(len(xint))
        C = (np.max(self.xint) - np.min(self.xint)) / 4 # Just calculate C once
        for i in range(len(xint)):
            for j in range(self.n):
                self.xint[j] /= (xint[j] - xint[i])
            new_weights[i] = 1 / np.prod([1 / C if (i + self.n) == j else (self.xint[i] - self.xint[j]) / C for j in range(self.n + len(xint))]) # Array broadcasting for equation 3
        self.weights = np.hstack([self.weights, new_weights])
        self.n += len(xint)


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    domain = np.linspace(-1, 1, 400)
    f = lambda x: 1/(1+25 * (x**2)) 
    plt.plot(domain, f(domain))
    normal_errors = []
    cheby_errors = []
    exponents = [2, 3, 4, 5, 6, 7, 8]
    for i in exponents:
        points = np.linspace(-1, 1, 2 ** i)
        poly = BarycentricInterpolator(points)
        poly.set_yi(f(points))
        f_hat = poly(domain)
        idk = np.max(np.abs(f(domain) - f_hat))
        normal_errors.append(idk)
        cheby_points = np.array([0.5 * ((-2) * np.cos(j * np.pi / 2 ** i)) for j in range(2 ** i)])
        new_poly = BarycentricInterpolator(cheby_points)
        new_poly.set_yi(f(cheby_points))
        new_f_hat = new_poly(domain)
        new_idk = np.max(np.abs(f(domain) - new_f_hat))
        cheby_errors.append(new_idk)
    plt.plot(exponents, normal_errors, color='red')
    plt.plot(exponents, cheby_errors, color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("prob5")


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    coeffs = np.zeros(n + 1)
    cheby_points = np.array([0.5 * ((-2) * np.cos(j * np.pi / n)) for j in range(n+1)])
    fft_part = np.real(np.fft.fft(f(cheby_points))) / (2 * n)
    for k in range(len(cheby_points)):
        gamma_k = 1 if k // n == 0 else 2
        coeffs[k] = gamma_k * fft_part[k]
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load("airdata.npy")
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)

    poly = BarycentricInterpolator(domain[temp2], data[temp2])
    plt.plot(domain, data, label='Observed points')
    plt.plot(domain, poly(domain), label='Interpolated points')
    plt.legend()
    plt.savefig("prob7.png")