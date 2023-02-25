import numpy as np
from scipy.integrate import quad
from scipy.integrate import fixed_quad # both from https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.fixed_quad.html
from scipy.special.orthogonal import p_roots # https://math.stackexchange.com/questions/3025013/gauss-quadrature-in-numerical-methods

# Defines the function to be integrated
def f(x):
    return x**3 + 2*x + 1

# Defines the number of evaluation points
n_points = 50

# Defines the integration limits
a = -1
b = 1

I_analytical = quad(f, a, b)[0] # computes integral analytically https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html

# Defines a range of sub-intervals to test
n_intervals_range = np.arange(10, 201, 10) # https://numpy.org/doc/stable/reference/generated/numpy.arange.html

# Initializing arrays to store the integration results and errors https://www.edureka.co/blog/arrays-in-python/ and https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
I_fixed = np.zeros_like(n_intervals_range, dtype=float)
I_gauss = np.zeros_like(n_intervals_range, dtype=float)
error_fixed = np.zeros_like(n_intervals_range, dtype=float)
error_gauss = np.zeros_like(n_intervals_range, dtype=float)

# Computes the integration results and errors for each value of interval
for i, n_intervals in enumerate(n_intervals_range): #https://www.geeksforgeeks.org/enumerate-in-python/

    # Fixed interval sizes method
    dx = (b-a)/n_intervals
    x_values = np.linspace(a, b, n_intervals+1)
    y_values = f(x_values)
    I_fixed[i] = dx*np.sum(y_values[:-1] + y_values[1:])/2
    error_fixed[i] = np.abs(I_fixed[i] - I_analytical)

    # Gaussian quadrature method
    x, w = p_roots(n_intervals)
    x_mapped = 0.5*(b-a)*x + 0.5*(b+a)
    y_values = f(x_mapped)
    I_gauss[i] = 0.5*(b-a)*np.sum(w*y_values)
    error_gauss[i] = np.abs(I_gauss[i] - I_analytical)

# Print the results
print("Analytical integral:", I_analytical)

for i, n_intervals in enumerate(n_intervals_range):  # https://engcourses-uofa.ca/books/numericalanalysis/numerical-integration/gauss-quadrature/
    print("Amount of sub-intervals:", n_intervals)
    print("Fixed interval sizes:   I = {:.6f}, error = {:.6f}".format(I_fixed[i], error_fixed[i]))
    print("Gaussian quadrature:    I = {:.6f}, error = {:.6f}".format(I_gauss[i], error_gauss[i]))
