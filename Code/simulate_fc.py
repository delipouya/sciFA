import numpy as np
import pandas as pd
import functions_plotting as fplot
from scipy.special import erf

import functions_fc_match_classifier as fmatch 
import functions_metrics as fmet
import functions_simulation as fsim

####################################
#### Simulate factors and covariates ######
####################################
### TODO: Define a class for a factor_simulator

num_factors = 15 ### put this to 100-1000 
num_mixtures = 2 ## each gaussian represents a covariate level 
num_samples = 1000

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

len(sim_factors_list)
len(sim_factors_list[0])
fplot.plot_histogram(a_random_factor, 'normal distribution')

overlap_mat_flat = fsim.convert_matrix_list_to_vector(overlap_mat_list) ### flatten the overlap matrix list

### convert sim_factors_list to a numpy nd array with shape (num_samples, num_factors)
sim_factors_array = np.asarray(sim_factors_list).T
sim_factors_df = pd.DataFrame(sim_factors_array, columns=['factor'+str(i+1) for i in range(num_factors)])
factor_scores = sim_factors_array
covariate_vector = pd.Series(covariate_list[0])

####################################
#### Matching between factors and covariates ######
####################################

### calculate the mean importance of each covariate level
mean_importance_df = fmatch.get_mean_importance_all_levels(covariate_vector, factor_scores)
fplot.plot_all_factors_levels_df(mean_importance_df, 
                                 title='F-C Match: Feature importance scores', color='coolwarm')
all_covariate_levels = mean_importance_df.index.values

#################################### 
#### Evaluate overlap and match scores for a single factor
#################################### 
factor_index = 0
match_score_mat_meanImp = fsim.get_pairwise_match_score_matrix(mean_importance_df,factor_index)
overlap_mat = overlap_mat_list[factor_index]

### plot the scatter plot of the overlap and match scores
fsim.plot_scatter(overlap_mat.flatten(), match_score_mat_meanImp.flatten(), title='feature importance')

#################################### 
#### calculate the overlap and matching scores for all the factors
#################################### 
match_score_mat_meanImp_list = []

for i in range(num_factors): ## i is the factor index
    match_score_mat_meanImp_list.append(fsim.get_pairwise_match_score_matrix(mean_importance_df,i))
match_score_mat_meanImp_flat = fsim.convert_matrix_list_to_vector(match_score_mat_meanImp_list)

#### plot the scatter plot of the overlap and match scores
fsim.plot_scatter(overlap_mat_flat, match_score_mat_meanImp_flat, title='feature importance')


####################################
#### evaluating bimodality score using simulated factors ####
####################################

bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores)
bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores)
likelihood_ratio_scores = fmet.get_likelihood_ratio_test_all(factor_scores)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
dip_scores, pval_scores = fmet.get_dip_test_all(factor_scores)
kurtosis_scores = fmet.get_factor_kurtosis_all(factor_scores)
outlier_sum_scores = fmet.get_outlier_sum_statistic_all(factor_scores)

bimodality_metrics = ['bic_km', 'calinski_harabasz_km', 'davies_bouldin_km', 'silhouette_km', 'vrs_km', 'wvrs_km',
                      'bic_gmm', 'silhouette_gmm', 'vrs_gmm', 'wvrs_gmm', 'likelihood_ratio', 'bimodality_index',
                          'dip_score', 'dip_pval', 'kurtosis', 'outlier_sum']
bimodality_scores = [bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km,
                                    silhouette_scores_km, vrs_km, wvrs_km,
                                    bic_scores_gmm, silhouette_scores_gmm,
                                    vrs_gmm, wvrs_gmm, likelihood_ratio_scores, bimodality_index_scores,
                                    dip_scores, pval_scores, kurtosis_scores, outlier_sum_scores]


# plot the scatter plot of the overlap and each of the bimodality scores
for i in range(len(bimodality_metrics)):
    fsim.plot_scatter(overlap_mat_flat, bimodality_scores[i], title=bimodality_metrics[i])



#### Scaled variance
SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)
fsim.plot_scatter(overlap_mat_flat, SV_all_factors[0], title='SV - cov1')
fsim.plot_scatter(overlap_mat_flat, SV_all_factors[1], title='SV - cov2')

### label dependent factor metrics
ASV_all_arith = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='arithmetic')
ASV_all_geo = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='geometric')
fsim.plot_scatter(overlap_mat_flat, ASV_all_arith, title='ASV - arithmatic')
fsim.plot_scatter(overlap_mat_flat, ASV_all_geo, title='ASV - geometric')


### calculate diversity metrics
factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df) ## calculated for each factor in the importance matrix
factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix


##### this comparison is not reliable as the number of levels are too small (two) to calculate diversity metrics
fsim.plot_scatter(overlap_mat_flat, factor_simpson_meanimp, title='factor simpson - mean importance')
fsim.plot_scatter(overlap_mat_flat, factor_entropy_meanimp, title='factor entropy - mean importance')


#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)
fsim.plot_scatter(overlap_mat_flat, factor_variance_all, title='factor variance')


### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
AUC_all_factors_df = fmet.get_AUC_all_factors_df(factor_scores, covariate_vector)
AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
fplot.plot_all_factors_levels_df(AUC_all_factors_df_1,
                                    title='Homogeneity: 1-AUC scores', color='hot')
fsim.plot_scatter(overlap_mat_flat, AUC_all_factors_df_1.iloc[0,:], title='cov1 - 1-AUC')
fsim.plot_scatter(overlap_mat_flat, AUC_all_factors_df_1.iloc[1,:], title='cov2 - 1-AUC')


#### calculate the correlation between the overlap and all of the scores and save in a dataframe
corr_df = pd.DataFrame()
for i in range(len(bimodality_metrics)):
    corr_df[bimodality_metrics[i]] = [np.corrcoef(overlap_mat_flat, bimodality_scores[i])[0,1]]
corr_df['SV_cov1'] = [np.corrcoef(overlap_mat_flat, SV_all_factors[0])[0,1]]
corr_df['SV_cov2'] = [np.corrcoef(overlap_mat_flat, SV_all_factors[1])[0,1]]
corr_df['ASV_arith'] = [np.corrcoef(overlap_mat_flat, ASV_all_arith)[0,1]]
corr_df['ASV_geo'] = [np.corrcoef(overlap_mat_flat, ASV_all_geo)[0,1]]

## only include in case #covariate levels > 3 
#corr_df['factor_entropy_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_entropy_meanimp)[0,1]]
#corr_df['factor_simpon_meanImp'] = [np.corrcoef(overlap_mat_flat, factor_simpson_meanimp)[0,1]]

corr_df['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]
corr_df['1-AUC_cov1'] = [np.corrcoef(overlap_mat_flat, AUC_all_factors_df_1.iloc[0,:])[0,1]]
corr_df['1-AUC_cov2'] = [np.corrcoef(overlap_mat_flat, AUC_all_factors_df_1.iloc[1,:])[0,1]]

corr_df = corr_df.T
corr_df.columns = ['overlap']
corr_df = corr_df.sort_values(by='overlap', ascending=False)
corr_df

####################################