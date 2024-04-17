#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate a factor with multiple covariates and calculate the overlap between the covariates
and the matching scores between the covariates and the factor
calculate the correlation between the overlap and the matching scores
repeat the simulation nn times and calculate the average correlation between the overlap and all of the scores
save the results in a csv file
"""

import numpy as np
import functions_plotting as fplot
from scipy.special import erf
from itertools import chain
import functions_fc_match_classifier as fmatch 
import pandas as pd
import functions_metrics as fmet
import matplotlib.pyplot as plt
import functions_simulation as fsim

import time

np.random.seed(0)

num_sim_rounds = 100
#num_sim_rounds = 2
num_factors = 30
num_mixtures = 2 ## each gaussian represents a covariate level 
num_samples = 1000
#### perform the simulation n times and calculate the average correlation between the overlap and all of the scores
corr_df = pd.DataFrame()
corr_df_list = []

    
sim_factors_list = []
overlap_mat_list = []
covariate_list = []

for i in range(num_factors):
    a_random_factor, overlap_matrix, mu_list, sigma_list, p_list = fsim.get_simulated_factor_object(n=num_samples, num_mixtures=num_mixtures, 
                                                                                            mu_min=0, mu_max=None,
                                                                                            sigma_min=0.5, sigma_max=1, p_equals=True)  
    sim_factors_list.append(fsim.unlist(a_random_factor))
    overlap_mat_list.append(overlap_matrix)
    covariate_list.append(fsim.get_sim_factor_covariates(a_random_factor))

overlap_mat_flat = fsim.convert_matrix_list_to_vector(overlap_mat_list) ### flatten the overlap matrix list



#### visualize overlap_mat_flat list as a one vector heatmap without reshaping
plt.figure()
plt.imshow(np.asarray(overlap_mat_flat).reshape(1,-1), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Overlap matrix')
plt.show()


### convert sim_factors_list to a numpy nd array with shape (num_samples, num_factors)
sim_factors_array = np.asarray(sim_factors_list).T
sim_factors_df = pd.DataFrame(sim_factors_array, columns=['factor'+str(i+1) for i in range(num_factors)])
factor_scores = sim_factors_array
covariate_vector = pd.Series(covariate_list[0])

### calculate the mean importance of each covariate level
mean_importance_df = fmatch.get_mean_importance_all_levels(covariate_vector, factor_scores)
all_covariate_levels = mean_importance_df.index.values

#################################### 
#### calculate the overlap and matching scores for all the factors
#################################### 
match_score_mat_meanImp_list = []

for i in range(num_factors): ## i is the factor index
    match_score_mat_meanImp_list.append(fsim.get_pairwise_match_score_matrix(mean_importance_df,i))
match_score_mat_meanImp_flat = fsim.convert_matrix_list_to_vector(match_score_mat_meanImp_list)

silhouette_scores, calinski_harabasz_scores, wvrs,\
    davies_bouldin_scores = fmet.get_kmeans_scores(factor_scores, time_eff=False)


#silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores, time_eff=False)
#likelihood_ratio_scores = fmet.get_likelihood_ratio_test_all(factor_scores)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
dip_scores, pval_scores = fmet.get_dip_test_all(factor_scores)
#kurtosis_scores = fmet.get_factor_kurtosis_all(factor_scores)
#outlier_sum_scores = fmet.get_outlier_sum_statistic_all(factor_scores)

bimodality_metrics = ['calinski_harabasz', 'davies_bouldin',
                        'silhouette', 'wvrs', 'bimodality_index', 'dip_score']
bimodality_scores = [ calinski_harabasz_scores, davies_bouldin_scores,
                                    silhouette_scores, wvrs,
                                    bimodality_index_scores,
                                    dip_scores]

#### Scaled variance
SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)

### visualize the SV_all_factors as a heatmap without a function
plt.figure()
plt.imshow(SV_all_factors, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Scaled variance of all factors')
plt.show()


### label dependent factor metrics - Homogeneity
ASV_all_arith = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='arithmetic')
ASV_all_geo = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='geometric')
ASV_simpson = fmet.get_all_factors_simpson(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)))
ASV_entropy = fmet.get_factor_entropy_all(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)))

### show heatmap of fmet.get_factors_SV_all_levels(factor_scores, covariate_vector))
plt.figure()
plt.imshow(fmet.get_factors_SV_all_levels(factor_scores, covariate_vector), cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Scaled variance of all factors per covariate')
plt.show()


### concatenrate all the ASV scores in a dataframe and show as a heatmap
ASV_all_df = pd.DataFrame([ASV_all_arith, ASV_all_geo, ASV_simpson, ASV_entropy])
ASV_all_df.index = ['ASV_arith', 'ASV_geo', 'ASV_simpson', 'ASV_entropy']
plt.figure()
plt.imshow(ASV_all_df, cmap='hot', interpolation='nearest')
plt.yticks(np.arange(ASV_all_df.shape[0]), ASV_all_df.index)
plt.colorbar()
plt.title('ASV scores')
plt.show()

### concatenrate all the ASV scores in a dataframe and show as a heatmap 

### scale the ASV_all_df so that each row has mean 0 and std 1
ASV_all_df_scaled = (ASV_all_df - ASV_all_df.mean(axis=1).values.reshape(-1,1))/ASV_all_df.std(axis=1).values.reshape(-1,1)
plt.figure()
plt.imshow(ASV_all_df_scaled, cmap='hot', interpolation='nearest')
plt.yticks(np.arange(ASV_all_df_scaled.shape[0]), ASV_all_df_scaled.index)
plt.colorbar()
plt.title('ASV scores scaled')
plt.show()


### show scatter plot of ASV_all_arith with overlap_mat_flat
plt.figure()
plt.scatter(overlap_mat_flat, ASV_all_arith)
plt.xlabel('overlap')
plt.ylabel('ASV_all_arith')
plt.title('scatter plot of overlap and ASV_all_arith')
plt.show()

####show scatter plot of scaled ASV (all four) with overlap_mat_flat in a loop
for i in range(ASV_all_df_scaled.shape[0]):
    plt.figure()
    plt.scatter(overlap_mat_flat, ASV_all_df_scaled.iloc[i])
    plt.xlabel('overlap')
    plt.ylabel(ASV_all_df_scaled.index[i])
    plt.title('scatter plot of overlap and '+ASV_all_df_scaled.index[i])
    plt.show()



### calculate diversity metrics (specificity)
factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
factor_simpson_meanimp = fmet.get_all_factors_simpson_D_index(mean_importance_df) ## calculated for each factor in the importance matrix
factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

### concatenrate all the diversity scores in a dataframe and show as a heatmap
diversity_all_df = pd.DataFrame([factor_simpson_meanimp, factor_entropy_meanimp])
diversity_all_df.index = ['factor_simpson_meanimp', 'factor_entropy_meanimp']
plt.figure()
plt.imshow(diversity_all_df, cmap='hot', interpolation='nearest')
plt.yticks(np.arange(diversity_all_df.shape[0]), diversity_all_df.index)
plt.colorbar()
plt.title('diversity scores')
plt.show()

### scale the diversity_all_df so that each row ranges from 0 to 1
diversity_all_df_scaled = (diversity_all_df - diversity_all_df.min(axis=1).values.reshape(-1,1))/(diversity_all_df.max(axis=1) - diversity_all_df.min(axis=1)).values.reshape(-1,1)
plt.figure()
plt.imshow(diversity_all_df_scaled, cmap='hot', interpolation='nearest')
plt.yticks(np.arange(diversity_all_df_scaled.shape[0]), diversity_all_df_scaled.index)
plt.colorbar()
plt.title('diversity scores scaled')
plt.show()


####splot scatter plot of factor_simpson_meanimp with overlap_mat_flat
plt.figure()
plt.scatter(overlap_mat_flat, factor_simpson_meanimp)
plt.xlabel('overlap')
plt.ylabel('factor_simpson_meanimp')
plt.title('scatter plot of overlap and factor_simpson_meanimp')
plt.show()


####splot scatter plot of factor_simpson_meanimp with overlap_mat_flat
plt.figure()
plt.scatter(overlap_mat_flat, factor_entropy_meanimp)
plt.xlabel('overlap')
plt.ylabel('factor_entropy_meanimp')
plt.title('scatter plot of overlap and factor_entropy_meanimp')
plt.show()

####show scatter plot of scaled diversity (all two) with overlap_mat_flat in a loop
for i in range(diversity_all_df_scaled.shape[0]):
    plt.figure()
    plt.scatter(overlap_mat_flat, diversity_all_df_scaled.iloc[i])
    plt.xlabel('overlap')
    plt.ylabel(diversity_all_df_scaled.index[i])
    plt.title('scatter plot of overlap and '+diversity_all_df_scaled.index[i])
    plt.show()

#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)

### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
AUC_all_factors_df = fmet.get_AUC_all_factors_df(factor_scores, covariate_vector)
AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
### calculate arithmatic and geometric mean of each factor using AUC_all_factors_df_1
AUC_1_arith_mean = fsim.get_arithmatic_mean_df(AUC_all_factors_df_1)
AUC_1_geo_mean = fsim.get_geometric_mean_df(AUC_all_factors_df_1)

#### calculate the correlation between the overlap and all of the scores and save in a dataframe
corr_df_temp = pd.DataFrame()
for i in range(len(bimodality_metrics)):
    corr_df_temp[bimodality_metrics[i]] = [np.corrcoef(overlap_mat_flat, bimodality_scores[i])[0,1]]


corr_df_temp['ASV_arith'] = [np.corrcoef(overlap_mat_flat, ASV_all_arith)[0,1]]
corr_df_temp['ASV_geo'] = [np.corrcoef(overlap_mat_flat, ASV_all_geo)[0,1]]

corr_df_temp['ASV_simpson'] = [np.corrcoef(overlap_mat_flat, ASV_simpson)[0,1]]
corr_df_temp['ASV_entropy'] = [np.corrcoef(overlap_mat_flat, ASV_entropy)[0,1]]

corr_df_temp['1-AUC_arith'] = [np.corrcoef(overlap_mat_flat, AUC_1_arith_mean)[0,1]]
corr_df_temp['1-AUC_geo'] = [np.corrcoef(overlap_mat_flat, AUC_1_geo_mean)[0,1]]

corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]

## only include in case #covariate levels > 3 - gini is a single value cant be saved in corr_df_temp
corr_df_temp['factor_entropy_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_entropy_meanimp)[0,1]]
corr_df_temp['factor_simpon_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_simpson_meanimp)[0,1]]

corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]
    
corr_df_temp = corr_df_temp.T
corr_df_temp.columns = ['overlap']
#corr_df_temp = corr_df_temp.sort_values(by='overlap', ascending=False)
corr_df_list.append(corr_df_temp)



end_time = time.time()
print('time: ', end_time - start_time)
### convert time to minutes
time_minutes = round((end_time - start_time)/60, 2)
print('time in minutes: ', time_minutes)

### calculate the average correlation between the overlap and all of the scores
corr_df = pd.concat(corr_df_list, axis=1)
### name the columns as overlap + column number
corr_df.columns = ['overlap_'+str(i) for i in range(corr_df.shape[1])]
### save as a csv file



bimodality_scores_df = pd.DataFrame(bimodality_scores).T
bimodality_scores_df.columns = bimodality_metrics
print(bimodality_scores_df.head())