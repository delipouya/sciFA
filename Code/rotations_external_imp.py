import numpy as np
import pandas as pd
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import statsmodels.multivariate.factor_rotation as factor_rotation
from factor_analyzer import FactorAnalyzer, Rotator
import rotations as rot
import functions_metrics as fmet
import functions_plotting as fplot
import functions_processing as fproc
import functions_fc_match_classifier as fmatch 
import rotations as rot
import constants as const


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
################

####################  using scmixology data
data = fproc.import_AnnData('/home/delaram/scLMM/sc_mixology/scMix_3cl_merged.h5ad')
y_cell_line, y_sample, y_protocol = fproc.get_metadata_scMix(data)
y, num_cells, num_genes = fproc.get_data_array(data)
y = fproc.get_sub_data(y)

colors_dict_scMix = fplot.get_colors_dict_scMix(y_protocol, y_cell_line)

print(y.shape)
print(y_sample.unique())
print(y_cell_line.unique())


### add protocol and cell line to the AnnData object
data.obs['protocol'] = y_protocol
data.obs['cell_line'] = y_cell_line

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


####################################
#### Running Varimax PCA on the data 
####################################
## apply varimax rotation to the loadings
#pca_scores_rot, loadings_rot, rotmat = rot.get_varimax_rotated_scores(pca_scores, factor_loading)

### check the rotation using allclose
### apply rotation by statsmodels package to the factor loadings and scores
### orthogonal rotation: quartimax, biquartimax, varimax, equamax, parsimax, parsimony
### oblique rotation: promax, oblimin, orthobimin, quartimin, biquartimin, varimin, equamin, parsimin


#### apply all the orthogonal rotations to the factor loadings and compare the results
rotations_ortho = ['quartimax', 'biquartimax', 'varimax', 'equamax', 'parsimax', 'parsimony']
loadings_rot_dict_ortho, rotmat_dict_ortho, scores_rot_dict_ortho = rot.get_rotation_dicts(factor_loading, rotations_ortho)

### plot the rotated scores
for rotation in rotations_ortho:
      fplot.plot_pca_scMix(scores_rot_dict_ortho[rotation], 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title= rotation + ' PCA of the data matrix')
      fplot.plot_pca_scMix(scores_rot_dict_ortho[rotation], 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title= rotation + ' PCA of the data matrix')
    

rot_corr_ortho = rot.get_rotation_corr_mat(loadings_rot_dict_ortho, num_factors=4)
## plot the correlation heatmap using seaborn
import seaborn as sns
sns.heatmap(rot_corr_ortho,
            xticklabels=rot_corr_ortho.columns,
            yticklabels=rot_corr_ortho.columns, cmap='coolwarm')


#rotations_oblique = ['promax', 'oblimin', 'orthobimin', 'quartimin', 'biquartimin', 'varimin', 'equamin', 'parsimin']
rotations_oblique = ['quartimin']
loadings_rot_dict_oblique, rotmat_dict_oblique, scores_rot_dict_oblique = rot.get_rotation_dicts(factor_loading, 
                                                                                             rotations_oblique,
                                                                                             factor_scores=pca_scores)
### plot the rotated scores
for rotation in rotations_oblique:
      fplot.plot_pca_scMix(scores_rot_dict_oblique[rotation], 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title= rotation + ' PCA of the data matrix')
      fplot.plot_pca_scMix(scores_rot_dict_oblique[rotation], 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title= rotation + ' PCA of the data matrix')
      
      