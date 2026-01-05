# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name>
<Class>
<Date>
"""

import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x, y = sy.symbols("x, y")
    return ((sy.Rational(2, 5)) * sy.exp(x ** 2 - y) * sy.cosh(x+y)) + ((3/7) * sy.log((x * y) + 1))


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x, j, i = sy.symbols("x, j, i")
    return sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5))


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-2,2]. Plot e^(-y^2) over the same domain for comparison.
    """
    domain = np.linspace(-2, 2, 1000)
    x, y, n = sy.symbols("x, y, n")
    expression = sy.summation(x**n / sy.factorial(n), (n, 0, N)) # The summation thing
    subbed = expression.subs(x, -y**2)
    mac = sy.lambdify(y, subbed, "numpy") # Maclauren series
    true_func = sy.lambdify(y, sy.exp(-y**2), "numpy") # With the y^2 plugged in

    # Time to plot
    plt.plot(domain, mac(domain), label=f"Maclaurin up to N={N}")
    plt.plot(domain, true_func(domain), label="e^(-y^2)")
    plt.legend()
    plt.savefig("prob3.png")

# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    x, y, r, theta = sy.symbols("x, y, r, theta") # Lots of stuff
    expression = 1 - ((x**2 + y**2)**(sy.Rational(7, 2)) + (18 * (x**5) * y) - (60 * (x ** 3) * (y ** 3)) + (18 * (y**5) * x) )/((x**2 + y ** 2)**3) # Copying equation 2
    with_x = expression.subs(x, r * sy.cos(theta)) # I don't know how to substitute twice at once
    with_y = with_x.subs(y, r * sy.sin(theta))
    new_expression = sy.simplify(with_y) # Simplified with substitutions
    r = sy.lambdify(theta, sy.solve(new_expression)[0][r]) # I'm unclear what this means

    domain = np.linspace(0, 2 * np.pi, 500)
    plt.plot(r(domain) * np.cos(domain), r(domain) * np.sin(domain))
    plt.savefig("prob4.png")


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    x, y, lam = sy.symbols('x y lam')
    A_minus_lambda = sy.Matrix([[x-y-lam,  x, 0],
               [x,  x-y-lam, x],
               [0, x, x-y-lam]])
    
    charpoly = sy.factor(A_minus_lambda.det()) # Characteristic polynomial
    lambdas = sy.solve(sy.Eq(charpoly, 0), lam) # It's super unclear which methods I'm allowed to use or not

    eig_dict = {} # Initialize for returning
    for val in lambdas:
        M_sub = A_minus_lambda.subs(lam, val) # Putting each lambda in
        ns = M_sub.nullspace()
        if not ns: # Just in case
            vec = sy.Matrix([0, 0, 0])
        else:
            vec = sy.Matrix([sy.simplify(v) for v in ns[0]]) # Just break down the vector in ns
        eig_dict[sy.simplify(val)] = vec

    return eig_dict


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points over [-5,5]. Determine which
    points are maxima and which are minima. Plot the maxima in one color and the
    minima in another color. Return the minima and maxima (x values) as two
    separate sets.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols("x")
    domain = np.linspace(-5, 5, 100)
    p = 2 * (x**6) - (51 * x**4) + (48 * x**3) + (312 * x**2) - (576 * x) - 100
    p_lam = sy.lambdify(x, p)
    p_prime = sy.Derivative(p, x)
    p_double_prime = sy.Derivative(p_prime, x)
    critical_points = sy.solve(p_prime.simplify(), x)
    min_vs_max = [p_double_prime.subs(x, c).simplify() for c in critical_points]
    plt.plot(domain, p_lam(domain))
    
    minima = set()
    maxima = set()
    for i in range(len(critical_points)):
        if min_vs_max[i] > 0:
            minima.add(critical_points[i])
        else:
            maxima.add(critical_points[i])

    plt.scatter(list(minima), [p_lam(x) for x in minima], color="blue", marker="o", label="Minima")
    plt.scatter(list(maxima), [p_lam(x) for x in maxima], color="red", marker="o", label="Maxima")

    plt.savefig("prob6.png")
    return minima, maxima


# Problem 7
def prob7():
    """Calculate the volume integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    r = sy.symbols('r')
    expr = sy.Rational(4,7)*sy.pi*r**7

    I = sy.lambdify(r, expr, modules=['numpy']) # Need a callable version

    rs = np.linspace(0, 3, 400)
    vals = I(rs)

    plt.figure(figsize=(7,4.5))
    plt.plot(rs, vals)
    plt.xlabel('r')
    plt.ylabel('Integral I(r)')
    plt.title('Integral of (x^2+y^2+z^2)^2 over ball of radius r')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prob7.png")

    # Print symbolic expression and value at r=2
    val_r2 = sy.simplify(expr.subs(r, 2))
    return float(val_r2.evalf())


print(prob7())
