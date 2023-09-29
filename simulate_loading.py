import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rotations as rot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from numpy.random import multivariate_normal

######################################################################
########## Simulation using random sampling of loadings and creating a correlation matrix from it
######################################################################
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


######################################################################
########## Simulation by specifying the loading matrix and creating a correlation matrix from it
######################################################################
# Simulate data with n factors and p variables
n = 500  # Number of observations
p = 20   # Number of variables
n_factors = 3  # Number of factors

# Specify the loadings for each factor manually
loadings_per_factor = [0.8, 0.7, 0.6, 0.5, 0.4]  # Adjust as needed
len(loadings_per_factor) == n_factors

# Initialize the loading matrix with zeros
loading_matrix = np.zeros((p, n_factors))

# Populate the loading matrix with loadings for each factor
start_idx = 0
for i in range(n_factors):
    loading_value = loadings_per_factor[i]
    end_idx = start_idx + int(p / n_factors)
    if i == n_factors - 1:
        end_idx = p
    loading_matrix[start_idx:end_idx, i] = loading_value
    start_idx = end_idx

loading_matrix
# Generate random error variances
error_variances = np.random.uniform(0.5, 2.0, p)
# Create a random correlation matrix
cor_matrix = np.dot(loading_matrix, loading_matrix.T) + np.diag(error_variances)

# Generate data based on the correlation matrix
simulated_data = multivariate_normal(np.zeros(p), cor_matrix, n)
simulated_data.shape


######################################################################
### calculating factor loadings using PCA
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=factors))])
pca_scores = pipeline.fit_transform(simulated_data)
pca = pipeline.named_steps['pca']
factor_loading = pca.components_
factor_loading.shape
rotated_loading= rot.varimax_rotation(factor_loading)
