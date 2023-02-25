

###### 1)
import numpy as np
import matplotlib.pyplot as plt
import scipy 

# pdf is probability density function which describes probability dist of continuous random variable 

# this function accepts and sets paramerters for the gamma function, it is the expected probability function
# I found the beta pdf function on https://vitalflux.com/beta-distribution-explained-with-python-examples/
def beta_pdf(x, a=2, b=3):
 # probability distribution around paramerters a and b
    return x**(a-1) * (1-x)**(b-1) / (scipy.special.gamma(a) * scipy.special.gamma(b) / scipy.special.gamma(a+b))
# I found this function on https://stackoverflow.com/questions/40428188/gamma-function-in-python

# the second function rejects the sampling until there are none left
def rejection_sampling(target_pdf, proposal_pdf, c, n_samples):
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, c*proposal_pdf(x))
        if y <= target_pdf(x):
            samples.append(x)
    return samples


# The target PDF is the probability density function of the distribution that is desired for generating the random samples
# Determines if a sample is accepted or rejected
# The proposal PDF is the probability density function of the dist that is used for the samples. 
a, b = 2, 3
target_pdf = beta_pdf
proposal_pdf = lambda x: 1


# constant c for the gamma that is kind of like intergrating I think?
# bounding for the acception and rejection taking place
c = scipy.special.gamma(a) * scipy.special.gamma(b) / scipy.special.gamma(a+b)
# from same site as above

# Sample from target distribution using rejection sampling with all the parameters
n_samples = 1000
samples = rejection_sampling(target_pdf, proposal_pdf, c, n_samples)

# Visualize the results means generate a plot right?
x = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# I keep getting an error about dimensions when trying to plot the proposal pdf
#ax.plot(x, proposal_pdf(x), label='Proposal PDF')
ax.plot(x, target_pdf(x), label='Target PDF')
ax.hist(samples, bins=20, density=True, alpha=0.5, label='Samples')
ax.legend()
plt.show()





### 2)

# I will define a 2D epsilloid
# epsilloid formula: (x/a)^2 + (y/b)^2 = 1

# code will generate random samples using rejection sampling for uniform points on ellipsoid


import numpy as np
from scipy.optimize import minimize # from https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
import matplotlib.pyplot as plt

def ellipsoid_pdf(x, y, a, b): #pdf for epsilloid https://belglas.files.wordpress.com/2018/03/cpwp.pdf
    return 4*np.pi/(3*a*b) * np.sqrt((a**2 - x**2)*(b**2 - y**2))
                                       

# the bounds for the ellipsoid 
def bounding_box_volume(x_min, x_max, y_min, y_max):
    """Volume of the bounding box."""
    return (x_max - x_min) * (y_max - y_min) 


def rejection_sampling(target_pdf, proposal_pdf, c, n_samples, x_min, x_max, y_min, y_max): 
    """Rejection sampling algorithm."""
    samples = []
    while len(samples) < n_samples:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        u = np.random.uniform(0, c*proposal_pdf(x, y))
        if u <= target_pdf(x, y):
            samples.append((x,y))
    return samples

# Define ellipsoid parameters and bounding box
a, b, c = 3, 2, 1
x_min, x_max = -a, a
y_min, y_max = -b, b

target_pdf = ellipsoid_pdf
proposal_pdf = lambda x,y: 1/bounding_box_volume   #distribution of the random points of the bounds