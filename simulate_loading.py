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
p = 70   # Number of variables - genes
factors = 4  # Number of factors

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



## TODO: how to calculate scores matrix for the simulated loadings
# https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
# multiple the data with the transposed pseudo-inverse of the rotated loadings: standardized scores

######################################################################
########## Simulation by specifying the loading matrix and creating a correlation matrix from it
######################################################################
# Simulate data with n factors and p variables
n = 1000  # Number of observations
p = 50   # Number of variables
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
factor_scores = pca_scores

rotation_results_varimax = rot.varimax_rotation(factor_loading.T)
rotation_results_promax = rot.promax_rotation(factor_loading.T)


### check if the rotloading of the varimax rotation is the same as the loadings_rot of the promax rotation
np.allclose(rotation_results_varimax['rotloading'], 
            rotation_results_promax['rotloading'])

rotation_results = rotation_results_promax
rotloading = rotation_results['rotloading']
rotmat = rotation_results['rotmat']
scores_rot = np.dot(factor_scores, rotmat)

print(rotloading.shape)
print(rotmat.shape)
print(factor_loading.shape)
print(factor_scores.shape)
print(scores_rot.shape)

#### plot the pca scores and the rotated pca scores
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(factor_scores[:,0], factor_scores[:,1])
plt.title('PCA scores')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.subplot(1, 2, 2)
plt.scatter(scores_rot[:,0], scores_rot[:,1])
plt.title('Rotated PCA scores')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#### plot the pca loadings and the rotated pca loadings and the matrix of the loadings
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.scatter(factor_loading.T[:,0], factor_loading.T[:,1])
plt.title('PCA loadings')
plt.xlabel('PC')
plt.ylabel('Variables')
plt.subplot(1, 3, 2)
plt.scatter(rotloading[:,0], rotloading[:,1])
plt.title('Rotated PCA loadings')
plt.xlabel('PC')
plt.ylabel('Variables')
plt.subplot(1, 3, 3)
plt.scatter(loading_matrix[:,0], loading_matrix[:,1])
plt.title('True loadings')
plt.xlabel('PC')
plt.ylabel('Variables')
plt.show()