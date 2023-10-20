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

num_sim_rounds = 20
num_factors = 15
num_mixtures = 2 ## each gaussian represents a covariate level 
num_samples = 10000
#### perform the simulation nn times and calculate the average correlation between the overlap and all of the scores
corr_df = pd.DataFrame()
corr_df_list = []

### calculate the time of the simulation
start_time = time.time()


for i in range(num_sim_rounds):
    
    start_time_in = time.time()
    print('simulation: ', i)
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

    ### convert sim_factors_list to a numpy nd array with shape (num_samples, num_factors)
    sim_factors_array = np.asarray(sim_factors_list).T
    sim_factors_df = pd.DataFrame(sim_factors_array, columns=['factor'+str(i+1) for i in range(num_factors)])
    factor_scores = sim_factors_array
    covariate_vector = pd.Series(covariate_list[0])

    ### calculate the mean importance of each covariate level
    mean_importance_df = fmatch.get_mean_importance_all_levels(covariate_vector, factor_scores)
    all_covariate_levels = mean_importance_df.index.values

    #### AUC score
    #### calculate the AUC of all the factors for all the covariate levels
    AUC_all_factors_df, wilcoxon_pvalue_all_factors_df = fmet.get_AUC_all_factors_df(factor_scores, covariate_vector)
    ### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
    AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)

    ### calculate arithmatic and geometric mean of each factor using AUC_all_factors_df_1
    AUC_1_arith_mean = fsim.get_arithmatic_mean_df(AUC_all_factors_df_1)
    AUC_1_geo_mean = fsim.get_geometric_mean_df(AUC_all_factors_df_1)

    #################################### 
    #### calculate the overlap and matching scores for all the factors
    #################################### 
    ### TODO: match score matrix seems not useful - consider removing it?
    match_score_mat_AUC_list = []
    match_score_mat_meanImp_list = []

    for i in range(num_factors): ## i is the factor index
        match_score_mat_AUC_list.append(fsim.get_pairwise_match_score_matrix(AUC_all_factors_df,i))
        match_score_mat_meanImp_list.append(fsim.get_pairwise_match_score_matrix(mean_importance_df,i))

    match_score_mat_AUC_flat = fsim.convert_matrix_list_to_vector(match_score_mat_AUC_list)
    match_score_mat_meanImp_flat = fsim.convert_matrix_list_to_vector(match_score_mat_meanImp_list)
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


    #### Scaled variance
    SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)

    ### label dependent factor metrics
    ASV_all_arith = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='arithmetic')
    ASV_all_geo = fmet.get_ASV_all(factor_scores, covariate_vector, mean_type='geometric')

    #### label free factor metrics
    factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
    factor_variance_all = fmet.get_factor_variance_all(factor_scores)

    ### calculate specificity
    factor_specificity_meanimp = fmet.get_all_factors_specificity(mean_importance_df)
    factor_specificity_AUC = fmet.get_all_factors_specificity(AUC_all_factors_df)


    #### calculate the correlation between the overlap and all of the scores and save in a dataframe
    corr_df_temp = pd.DataFrame()
    for i in range(len(bimodality_metrics)):
        corr_df_temp[bimodality_metrics[i]] = [np.corrcoef(overlap_mat_flat, bimodality_scores[i])[0,1]]

    
    corr_df_temp['ASV_arith'] = [np.corrcoef(overlap_mat_flat, ASV_all_arith)[0,1]]
    corr_df_temp['ASV_geo'] = [np.corrcoef(overlap_mat_flat, ASV_all_geo)[0,1]]
    
    corr_df_temp['1-AUC_arith'] = [np.corrcoef(overlap_mat_flat, AUC_1_arith_mean)[0,1]]
    corr_df_temp['1-AUC_geo'] = [np.corrcoef(overlap_mat_flat, AUC_1_geo_mean)[0,1]]

    corr_df_temp['specificity_meanimp'] = [np.corrcoef(overlap_mat_flat, factor_specificity_meanimp)[0,1]]
    corr_df_temp['specificity_AUC'] = [np.corrcoef(overlap_mat_flat, factor_specificity_AUC)[0,1]]

    corr_df_temp['factor_variance'] = [np.corrcoef(overlap_mat_flat, factor_variance_all)[0,1]]
    corr_df_temp['factor_entropy'] = [np.corrcoef(overlap_mat_flat, factor_entropy_all)[0,1]]
    
    corr_df_temp = corr_df_temp.T
    corr_df_temp.columns = ['overlap']
    #corr_df_temp = corr_df_temp.sort_values(by='overlap', ascending=False)
    corr_df_list.append(corr_df_temp)

    end_time_in = time.time()
    print('simulation: ', str(i), ' time: ', end_time_in - start_time_in)


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
corr_df.to_csv('metric_overlap_corr_df_sim'+str(num_sim_rounds)+'_v2.csv')