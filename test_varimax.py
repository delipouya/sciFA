import numpy as np
import pandas as pd
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import statsmodels.multivariate.factor_rotation as factor_rotation

import rotations as rot

loading = np.array([[0.8, 0.6, 0.0, 0.0, 0.0, 0.0],
                    [0.6, 0.8, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.8, 0.6, 0.0, 0.0],
                    [0.0, 0.0, 0.6, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.8, 0.6],
                    [0.0, 0.0, 0.0, 0.0, 0.6, 0.8]])
rotation_type = 'varimax'
loadings_rot, rotmat = factor_rotation.rotate_factors(loading, 
                                                      method=rotation_type,
                                                      algorithm_kwargs=['gpa'],
                                                      ) #max_triesint=max_triesint, tol=tol 



#print(np.allclose(dot(loading, rotmat), loadings_rot))
####TODO: check if rotated scores are calculated as scale(original pc scores) %*% rotmat
#scores_rot = dot(factor_scores, rotmat)  

#A = np.random.randn(10,3)
L, T = factor_rotation.rotate_factors(A,'varimax')
np.allclose(L,A.dot(T))
### save A to a csv file
np.savetxt('A.csv', A, delimiter=',')

manual_implementation = rot.varimax(A)

'''
V1     V2     V3    
 [1,]         1.907 -1.011
 [2,] -0.873  0.910 -2.867
 [3,] -0.282 -2.052 -0.226
 [4,]  0.632  0.568  0.837
 [5,] -2.541 -0.418 -0.779
 [6,]        -0.702 -0.157
 [7,]  0.961              
 [8,]  0.595        -1.004
 [9,]  0.602 -0.505 -0.289
[10,]  0.562  0.963  2.445


          [,1]       [,2]       [,3]
[1,]  0.8930460  0.4470581 0.05106825
[2,] -0.4372999  0.8355626 0.33257169
[3,]  0.1060081 -0.3193339 0.94169427



function (x, normalize = TRUE, eps = 1e-05) 
{
  nc <- ncol(x)
  if (nc < 2) 
    return(x)
  if (normalize) {
    sc <- sqrt(drop(apply(x, 1L, function(x) sum(x^2))))
    x <- x/sc
  }
  p <- nrow(x)
  TT <- diag(nc)
  d <- 0
  for (i in 1L:1000L) {
    z <- x %*% TT
    B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
    sB <- La.svd(B)
    TT <- sB$u %*% sB$vt
    dpast <- d
    d <- sum(sB$d)
    if (d < dpast * (1 + eps)) 
      break
  }
  z <- x %*% TT
  if (normalize) 
    z <- z * sc
  dimnames(z) <- dimnames(x)
  class(z) <- "loadings"
  list(loadings = z, rotmat = TT)
}

'''