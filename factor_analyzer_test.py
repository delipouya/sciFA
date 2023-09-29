import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, Rotator
import rotations as rot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


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

##########
### calculating factor loadings using PCA
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=factors))])
pca_scores = pipeline.fit_transform(simulated_data)
pca = pipeline.named_steps['pca']
factor_loading = pca.components_
factor_loading.shape
rotated_loading= varimax_rotation(factor_loading)



method = 'varimax'
loading_list = []
fa = FactorAnalyzer(rotation=None)
fa.fit(simulated_data)
rotator = Rotator(method=method)
rotated_loading = rotator.fit_transform(fa.loadings_) #
loading_list.append(rotated_loading)


#### using factor analyzer package to calculate the rotations
ortho_rots = ['varimax', 'oblimax', 'quartimax', 'equamax']
oblique_rots = ['promax', 'oblimin', 'quartimin']
rotation_methods = ortho_rots + oblique_rots

# Loop through each rotation method and assess the results
for method in rotation_methods:
    # Perform factor analysis with the current rotation method
    
    fa = FactorAnalyzer(rotation=None)
    fa.fit(simulated_data)
    rotator = Rotator(method=method)
    rotated_loading = rotator.fit_transform(fa.loadings_) 
    loading_list.append(rotated_loading)

# https://factor-analyzer.readthedocs.io/en/latest/_modules/factor_analyzer/rotator.html

