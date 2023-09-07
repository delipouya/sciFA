
import numpy as np
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd


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

    return {'rotloadings':Lambda, 'rotmat':R}



##### First implementation of Varimax rotation ####
## method: scale(original pc scores) %*% rotmat
## rotmat: the rotation matrix
def get_varimax_rotated_scores(factor_scores, loading):
    '''
    apply varimax rotation to the factor scores
    factor_scores: the factor scores matrix
    loading: the factor loading matrix
    '''
    varimax_rot = varimax(loading.T)
    loadings_rot = varimax_rot['rotloadings']
    rotmat = varimax_rot['rotmat']
    scores_rot = dot(factor_scores, rotmat)
    
    return scores_rot, loadings_rot, rotmat
