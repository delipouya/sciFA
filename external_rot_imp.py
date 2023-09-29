import numpy as np
import pandas as pd
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import statsmodels.multivariate.factor_rotation as factor_rotation
from factor_analyzer import FactorAnalyzer, Rotator
import rotations as rot

### implementation avaible on https://stackoverflow.com/questions/44956947/how-to-use-varimax-rotation-in-python
def varimax(Phi, gamma = 1.0, q = 100, tol = 1e-6):
    '''
    varimax rotation
    Phi: the factor loading matrix
    gamma: the power of the objective function
    q: the maximum number of iterations
    tol: the tolerance for convergence
    '''
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    Lambda = dot(Phi, R)

    return {'rotloading':Lambda, 'rotmat':R}


################## simulation of data
# Simulate data with 3 factors and 10 variables
np.random.seed(123)
n = 1000  # Number of observations
p = 3000   # Number of variables - genes
factors = 10  # Number of factors

# Create a random correlation matrix with 3 factors
loading_matrix = np.random.randn(p, factors)
# Generate random error variances
error_variances = np.random.uniform(0.5, 2.0, p)
# Create a random correlation matrix
cor_matrix = np.dot(loading_matrix, loading_matrix.T) + np.diag(error_variances)

# Generate data based on the correlation matrix
##(number of observations, number of variables)
simulated_data = np.random.multivariate_normal(mean=np.zeros(p), cov=cor_matrix, size=n)  
simulated_data.shape
####################################


####################################
#### using factor analyzer package to calculate the rotations
# source code: https://factor-analyzer.readthedocs.io/en/latest/_modules/factor_analyzer/rotator.html

ortho_rots = ['varimax', 'oblimax', 'quartimax', 'equamax']
oblique_rots = ['promax', 'oblimin', 'quartimin']
rotation_methods = ortho_rots + oblique_rots
loading_list = []
# Loop through each rotation method and assess the results
for method in rotation_methods:
    # Perform factor analysis with the current rotation method
    
    fa = FactorAnalyzer(rotation=None)
    fa.fit(simulated_data)
    rotator = Rotator(method=method)
    rotated_loading = rotator.fit_transform(fa.loadings_) 
    loading_list.append(rotated_loading)


#######################################
## using the varimax rotation from statsmodels package
## results was not consistent with base R results tested in R
# loading matrix did not seem to have a simpler structure post rotation
rotation_type = 'varimax'
loadings_rot, rotmat = factor_rotation.rotate_factors(loading_matrix, 
                                                      method=rotation_type,
                                                      algorithm_kwargs=['gpa'],
                                                      ) #max_triesint=max_triesint, tol=tol 
