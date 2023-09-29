
import numpy as np
import pandas as pd
from numpy import eye, asarray, dot, sum, diag
#from numpy.linalg import svd
import statsmodels.multivariate.factor_rotation as factor_rotation
import numpy as np
from scipy.linalg import svd


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


##########  implementation of base R Varimax rotation in python ##########
def varimax_rotation(x, normalize=True, eps=1e-05):
    nc = x.shape[1]
    if nc < 2:
        return x
    
    if normalize:
        sc = np.sqrt(np.sum(x**2, axis=1))
        x = x / sc[:, np.newaxis]
    
    p = x.shape[0]
    TT = np.eye(nc)
    d = 0
    
    for i in range(1, 1001):
        z = np.dot(x, TT)
        B = np.dot(x.T, z**3 - np.dot(z, np.diag(np.sum(z**2, axis=0)) / p))
        u, sB, vh = svd(B, full_matrices=False)
        TT = np.dot(u, vh)
        dpast = d
        d = np.sum(sB)
        
        if d < dpast * (1 + eps):
            break
    
    z = np.dot(x, TT)
    
    if normalize:
        z = z * sc[:, np.newaxis]
    
    return {'rotloading': z, 'rotmat': TT}



def promax_rotation(x, m=4, normalize=True, eps=1e-05):
    if x.shape[1] < 2:
        return x
    
    # Varimax rotation
    xx = varimax_rotation(x, normalize)
    x = xx['rotloading']
    
    # Calculate Q matrix
    q = x * np.abs(x)**(m - 1)
    
    try:
        # Perform linear regression
        u = np.linalg.lstsq(x, q, rcond=None)[0]
        d = np.diag(np.linalg.inv(np.dot(u.T, u)))
        u = np.dot(u, np.diag(np.sqrt(d)))
    except LinAlgError:
        return x  # Return the original loadings if there is an error
    
    # Calculate the rotated loadings
    z = np.dot(x, u)
    u = np.dot(xx['rotmat'], u)
    
    return {'loadings': z, 'rotmat': u}

# Example usage:
# Assuming 'x' is your data matrix
# z = promax_rotation(x)



##### First implementation of Varimax rotation ####
## method: scale(original pc scores) %*% rotmat
## rotmat: the rotation matrix
def get_varimax_rotated_scores(factor_scores, loading):
    '''
    apply varimax rotation to the factor scores
    factor_scores: the factor scores matrix
    loading: the factor loading matrix
    '''
    rotation_results  = varimax_rotation(loading)
    #loadings_rot, rotmat  = varimax(loading)
    loadings_rot, rotmat = rotation_results['rotloading'], rotation_results['rotmat']
    scores_rot = dot(factor_scores, rotmat)
    
    return scores_rot, loadings_rot, rotmat



### apply rotation by statsmodels package to the factor loadings and scores

### orthogonal rotation: quartimax, biquartimax, varimax, equamax, parsimax, parsimony
### oblique rotation: promax, oblimin, orthobimin, quartimin, biquartimin, varimin, equamin, parsimin
def get_rotated_factors(loading, rotation_type, factor_scores=None, max_triesint=500, tol=1e-6):
    '''
    apply rotation to the factor scores
    factor_scores: the factor scores matrix
    loading: the factor loading matrix
    rotation_type: the type of rotation
    '''
    if factor_scores is None:
        factor_scores = eye(loading.shape[0])
    
    print('rotation type: ', rotation_type)
    loadings_rot, rotmat = factor_rotation.rotate_factors(loading, method=rotation_type) #max_triesint=max_triesint, tol=tol 
    #print(np.allclose(dot(loading, rotmat), loadings_rot))
    ####TODO: check if rotated scores are calculated as scale(original pc scores) %*% rotmat
    scores_rot = dot(factor_scores, rotmat)  
    
    return loadings_rot, rotmat, scores_rot



def get_rotation_dicts(factor_loading, rotations, factor_scores):
      '''
      get a dictionary of all the orthogonal rotations
      '''
      loadings_rot_dict = {}
      rotmat_dict = {}
      scores_rot_dict = {}
      for rotation in rotations:
            loadings_rot, rotmat, scores_rot =  get_rotated_factors(loading=factor_loading.T, 
                                                                     rotation_type=rotation, 
                                                                     factor_scores=factor_scores)

            loadings_rot_dict[rotation] = loadings_rot
            rotmat_dict[rotation] = rotmat
            scores_rot_dict[rotation] = scores_rot
      
      return loadings_rot_dict, rotmat_dict, scores_rot_dict


def get_rotation_corr_mat(loadings_rot_dict, num_factors) -> pd.DataFrame:
      '''
      get the correlation matrix of the loadings for all the rotations
      '''
      rotation_names = list(loadings_rot_dict.keys()) 
      first_rot = rotation_names[0]
      loadings_rot_df = pd.DataFrame(loadings_rot_dict[first_rot][:,0:num_factors], 
                                     columns=[first_rot + str(i) for i in range(1,num_factors+1)])
      for rotation in rotation_names[1:]:
            loadings_rot_df = pd.concat([loadings_rot_df, 
                                         pd.DataFrame(loadings_rot_dict[rotation][:,0:num_factors], 
                                                      columns=[str(rotation) + str(i) for i in range(1,num_factors+1)])], axis=1)
      corr = loadings_rot_df.corr()
      #corr.style.background_gradient(cmap='coolwarm')
      return corr
