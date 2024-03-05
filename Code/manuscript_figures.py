import sys
sys.path.append('./Code/')
import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.graphics.api import abline_plot

import functions_metrics as fmet
import functions_plotting as fplot
import functions_processing as fproc
import functions_fc_match_classifier as fmatch 
import functions_GLM as fglm

import rotations as rot
import constants as const
import statsmodels.api as sm

np.random.seed(10)

### make a random matrix and visualize it using heatmap colored by cmap='YlGnBu' without any function
random_matrix = np.random.rand(4, 12)
plt.figure(figsize=(34,9))
plt.imshow(random_matrix, cmap='YlGnBu')
## add row and column labels
plt.xticks(np.arange(random_matrix.shape[1]), 
           ['F'+str(i+1) for i in range(random_matrix.shape[1])], fontsize=45, rotation=45)
### increase legend font size. add ["Bimodality", "Specificity", "Effect size", "Homogeneity"] as the y axis labels
plt.yticks(np.arange(random_matrix.shape[0]), 
           ["Separability", "Specificity", "Effect size", "Homogeneity"], fontsize=60)
plt.colorbar()
### save as a pdf
plt.savefig('../Plots/random_matrix_metric_scores.pdf')
plt.show()



##### make a radom matrix with elements as 1, 0 and 0.5 with more zeros
###and visualize it using heatmap colored by coolwarm without any function
random_matrix = np.random.choice([0, 0.2, 0.7, 1], size=(5, 12), p=[1/2, 1/4, 1/8,1/8])

plt.figure(figsize=(34,9))
plt.imshow(random_matrix, cmap='coolwarm')
## add row and column labels
plt.xticks(np.arange(random_matrix.shape[1]), 
           ['F'+str(i+1) for i in range(random_matrix.shape[1])], fontsize=55, rotation=45)
### set y ticks as cov-1 to cov-4
plt.yticks(np.arange(random_matrix.shape[0]), 
           ['cov'+str(i+1) for i in range(random_matrix.shape[0])], fontsize=70)           
plt.colorbar()
### save as a pdf
plt.savefig('../Plots/random_matrix_match_coolwarm.pdf')
plt.show()



### make a bimodal distribution of two normal distributions 
random_numbers = np.random.normal(0, 1, 5000)
random_numbers = np.concatenate((random_numbers, np.random.normal(5, 1, 1000)))
#### visualize it using a histogram without a function with a white background
plt.figure(figsize=(10,5))
plt.hist(random_numbers, bins=100)
### add white background
plt.rcParams['axes.facecolor'] = 'white'
ax = plt.axes()
ax.set_facecolor('white')
# Border color:
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_color('black')
plt.xlabel('Feature importance scores', fontsize=30)
plt.ylabel('Frequency', fontsize=30)
### increase tick font size
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('', fontsize=20)
plt.savefig('../Plots/imp_score_histogram_fig1.pdf')
plt.show()


## make a barplot for the following vector and visualize it without any function
random_vector = np.array([2, 1, 1, 0, 2, 1, 0, 0 ,1 ,0])
plt.figure(figsize=(10,5))
plt.bar(np.arange(len(random_vector)), random_vector)
### add white background
plt.rcParams['axes.facecolor'] = 'white'
ax = plt.axes()
ax.set_facecolor('white')
# Border color:
for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_color('black')
plt.xlabel('Feature importance scores', fontsize=30)
plt.ylabel('Frequency', fontsize=30)
### set y ticks as 0, 1, 2
plt.yticks([0, 1, 2], fontsize=26)
## set x ticks as F1 to F10
plt.xticks(np.arange(len(random_vector)), ['F'+str(i+1) for i in range(len(random_vector))], fontsize=29)
plt.title('', fontsize=20)
plt.savefig('../Plots/imp_score_barplot_fig1.pdf')  
plt.show()
