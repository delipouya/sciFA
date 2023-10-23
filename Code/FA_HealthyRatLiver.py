import sys
sys.path.append('./Code/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import functions_metrics as fmet
import functions_plotting as fplot
import functions_processing as fproc
import functions_GLM as fglm
import functions_fc_match_classifier as fmatch 
import rotations as rot
import constants as const
## SET SEED
np.random.seed(0)


#### calculate the goodness of fit of GLM model
# https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html#statsmodels.genmod.generalized_linear_model.GLMResults

### statsmodels GLM source code:
# https://github.com/statsmodels/statsmodels/blob/main/statsmodels/genmod/families/links.py

### poisson family link function
# https://github.com/statsmodels/statsmodels/blob/main/statsmodels/genmod/families/family.py


data_file_path = './Data/inputdata_rat_set1_countData_2.h5ad'
data = fproc.import_AnnData(data_file_path)
y, num_cells, num_genes = fproc.get_data_array(data)
y_sample, y_strain, y_cluster = fproc.get_metadata_ratLiver(data)
y = fproc.get_sub_data(y, random=False) # subset the data to num_genes HVGs

### randomly subsample the cells to 6000 cells
subsample_index = np.random.choice(y.shape[0], size=3000, replace=False)
y = y[subsample_index,:]
data = data[subsample_index,:]
y_sample = y_sample[subsample_index]
y_strain = y_strain[subsample_index]
y_cluster = y_cluster[subsample_index]

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_RNA')

#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sample', data) 
x = np.column_stack((data.obs.nCount_RNA, x_sample)) 
x = sm.add_constant(x) ## adding the intercept

### fit GLM to each gene
glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)

####################################
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape #(factors, genes)

colors_dict_ratLiver = fplot.get_colors_dict_ratLiver(y_sample, y_strain, y_cluster)

plt.plot(pca.explained_variance_ratio_)

### make a dictionary of colors for each sample in y_sample
fplot.plot_pca(pca_scores, 4, cell_color_vec= colors_dict_ratLiver['cluster'], 
               title='PCA of pearson residuals - reg: library size')
fplot.plot_pca(pca_scores, 4, cell_color_vec= colors_dict_ratLiver['sample'])
fplot.plot_pca(pca_scores, 4, cell_color_vec= colors_dict_ratLiver['strain'])

####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'strain'#'cell_line'
covariate_level = 'DA'
#factor_scores = pca_scores
covariate_vector = y_strain

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])
fplot.plot_pca(pca_scores_varimax, 9, cell_color_vec= colors_dict_ratLiver['strain'])
fplot.plot_pca(pca_scores_varimax, 9, cell_color_vec= colors_dict_ratLiver['cluster'])

######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_ratLiver['strain'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_ratLiver['cluster'])

########################

factor_loading = varimax_loading #pca_loading
factor_scores = pca_scores_varimax #pca_scores_promax #pca_scores_varimax
### calculate the mean importance of each covariate level
mean_importance_df_strain = fmatch.get_mean_importance_all_levels(y_strain, factor_scores)
mean_importance_df_cluster = fmatch.get_mean_importance_all_levels(y_cluster, factor_scores)

### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
mean_importance_df = pd.concat([mean_importance_df_strain, mean_importance_df_cluster], axis=0)
mean_importance_df.shape
fplot.plot_all_factors_levels_df(mean_importance_df, title='F-C Match: Feature importance scores', color='coolwarm')
## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values

### save the mean_importance_df to a csv file
mean_importance_df.to_csv('mean_importance_df_hrl.csv')



##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level

### plot the histogram of all the values in mean importance scores
fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                     title='F-C Match: Feature importance scores') 

### choosing a threshold for the feature importance scores

threshold = 0.3
threshold = fmatch.get_otsu_threshold(mean_importance_df.values.flatten())
print(threshold)

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                        title='F-C Match: Feature importance scores', threshold=threshold)

matched_factor_dist, percent_matched_fact = fmatch.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched_cov = fmatch.get_percent_matched_covariate(mean_importance_df, threshold=threshold)

print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
fplot.plot_matched_factor_dist(matched_factor_dist)
fplot.plot_matched_covariate_dist(matched_covariate_dist, covariate_levels=all_covariate_levels)


#### AUC score
#### calculate the AUC of all the factors for all the covariate levels
AUC_all_factors_df_strain, wilcoxon_pvalue_all_factors_df_strain = fmet.get_AUC_all_factors_df(factor_scores, y_strain)
AUC_all_factors_df_cluster, wilcoxon_pvalue_all_factors_df_cluster = fmet.get_AUC_all_factors_df(factor_scores, y_cluster)

AUC_all_factors_df = pd.concat([AUC_all_factors_df_strain, AUC_all_factors_df_cluster], axis=0)
wilcoxon_pvalue_all_factors_df = pd.concat([wilcoxon_pvalue_all_factors_df_strain, wilcoxon_pvalue_all_factors_df_cluster], axis=0)

fplot.plot_all_factors_levels_df(AUC_all_factors_df, 
                                 title='F-C Match: AUC scores', color='inferno') #YlOrBr






####################################
##### Factor metrics #####
####################################
### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
## list of color maps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
fplot.plot_all_factors_levels_df(AUC_all_factors_df_1,
                                    title='Homogeneity: 1-AUC scores', color='viridis')

### p value score is not a good indicative of the homogeneity of the factors
#fplot.plot_all_factors_levels_df(wilcoxon_pvalue_all_factors_df, 
#                                 title='Homogeneity: Wilcoxon pvalue', color='rocket_r')

#### Specificity 
factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)
fplot.plot_metric_barplot(factor_specificity_all, 'Specificity of each factor')
#highest_vote_factor = a_cov_match_factor_dict[covariate_level]; i = int(highest_vote_factor[2:]) - 1
factor_i = 7
fplot.plot_factor_scatter(factor_scores, 0, factor_i, colors_dict_ratLiver['strain'])

#### Entropy 
factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
fplot.plot_metric_barplot(factor_entropy_all, 'Entropy of each factor')
fplot.plot_histogram(factor_scores[:,np.argmin(factor_entropy_all)], title='factor with the min entropy')

#### Variance
factor_variance_all = fmet.get_factor_variance_all(factor_scores)
fplot.plot_metric_barplot(factor_variance_all, 'Variance of each factor')
fplot.plot_factor_scatter(factor_scores, 0, np.argmin(factor_variance_all), colors_dict_ratLiver['strain'], 
                           title='factor with the minimum variance')

#### Scaled variance
SV_all_factors_strain = fmet.get_factors_SV_all_levels(factor_scores, y_strain)
SV_all_factors_cluster = fmet.get_factors_SV_all_levels(factor_scores, y_cluster)
SV_all_factors = np.concatenate((SV_all_factors_strain, SV_all_factors_cluster), axis=0)
#all_covariate_levels = np.concatenate((y_protocol.unique(), y_cell_line.unique()), axis=0)

### convert to SV_all_factors to dataframe
SV_all_factors_df = pd.DataFrame(SV_all_factors)
SV_all_factors_df.columns = AUC_all_factors_df.columns
SV_all_factors_df.index = AUC_all_factors_df.index
## scale each factor from 0 to 1
SV_all_factors_df = SV_all_factors_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

fplot.plot_all_factors_levels_df(SV_all_factors_df,
                                    title='Homogeneity: scaled variance scores', color='Reds')
fplot.plot_all_factors_levels(SV_all_factors, all_covariate_levels, 
                              title='Scaled variance for all the factors', color='RdPu')


ASV_sample_all = fmet.get_ASV_all(factor_scores, y_sample, mean_type='arithmetic')
ASV_strain_all = fmet.get_ASV_all(factor_scores, y_strain, mean_type='arithmetic')
ASV_cluster_all = fmet.get_ASV_all(factor_scores, y_cluster, mean_type='arithmetic')
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(ASV_cluster_all), colors_dict_ratLiver['cluster'], 
                           title='factor with maximum ASV of cluster')



#### bimodality scores

### kmeans clustering based bimodality metrics
### TODO: try initializing kmeans with the min and max factor scores for cluster centers
bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores)

fplot.plot_metric_barplot(silhouette_scores_km, 'Silhouette score of each factor')
fplot.plot_metric_barplot(calinski_harabasz_scores_km, 'Calinski Harabasz score of each factor')
fplot.plot_metric_barplot(wvrs_km, 'weighted variance ratio score of each factor')
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(silhouette_scores_km), colors_dict_ratLiver['strain'],
                            title='factor with maximum silhouette score')
fplot.plot_histogram(factor_scores[:,np.argmin(silhouette_scores_km)], 
                     title='factor with the minimum silhouette score')

### plot the correlation of the metrics
bimodality_metrics = ['bic_km', 'calinski_harabasz', 'davies_bouldin', 'silhouette', 'vrs', 'wvrs']
bimodality_scores = np.concatenate((bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km,
                                    silhouette_scores_km, vrs_km, wvrs_km), axis=0).reshape(len(bimodality_metrics), -1)
#fplot.plot_metric_correlation(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))



### gmm clustering based bimodality metrics #### model-based clustering fmet.
bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores)
fplot.plot_metric_barplot(silhouette_scores_gmm, 'Silhouette score of each factor')
fplot.plot_metric_barplot(bic_scores_gmm, 'BIC score of each factor')

### plot the correlation of the metrics
bimodality_metrics = ['bic', 'silhouette', 'vrs', 'wvrs']
bimodality_scores = np.concatenate((bic_scores_gmm, silhouette_scores_gmm, 
                                    vrs_gmm, wvrs_gmm), axis=0).reshape(len(bimodality_metrics), -1)
#fplot.plot_metric_correlation(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))

### likelihood test 
likelihood_ratio_scores = fmet.get_likelihood_ratio_test_all(factor_scores)
fplot.plot_metric_barplot(likelihood_ratio_scores, 'Likelihood ratio score of each factor')

### bimodality index metric
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
fplot.plot_metric_barplot(bimodality_index_scores, 'Bimodality index score of each factor')


### dip test based bimodality metrics
dip_scores, pval_scores = fmet.get_dip_test_all(factor_scores)
fplot.plot_metric_barplot(dip_scores, 'Dip test score of each factor')
fplot.plot_metric_barplot(pval_scores, 'Dip test p values of each factor')


#### kurtosis
kurtosis_scores = fmet.get_factor_kurtosis_all(factor_scores)
fplot.plot_metric_barplot(kurtosis_scores, 'Kurtosis score of each factor')

#### outlier sum statistic 
outlier_sum_scores = fmet.get_outlier_sum_statistic_all(factor_scores)
fplot.plot_metric_barplot(outlier_sum_scores, 'Outlier sum score of each factor')


### plot the correlation of all the bimodality metrics
bimodality_metrics = ['bic_km', 'calinski_harabasz_km', 'davies_bouldin_km', 'silhouette_km', 'vrs_km', 'wvrs_km',
                      'bic_gmm', 'silhouette_gmm', 'vrs_gmm', 'wvrs_gmm', 'likelihood_ratio', 'bimodality_index',
                          'dip_score', 'dip_pval', 'kurtosis', 'outlier_sum']
bimodality_scores = np.concatenate((bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km,
                                    silhouette_scores_km, vrs_km, wvrs_km,
                                    bic_scores_gmm, silhouette_scores_gmm,
                                    vrs_gmm, wvrs_gmm, likelihood_ratio_scores, bimodality_index_scores,
                                    dip_scores, pval_scores, kurtosis_scores, outlier_sum_scores), axis=0).reshape(len(bimodality_metrics), -1)
fplot.plot_metric_correlation_clustermap(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))
fplot.plot_metric_dendrogram(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))



####################################

#### label free factor metrics
factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
factor_variance_all = fmet.get_factor_variance_all(factor_scores)

### bimoality metrics
bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores)
bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores)
likelihood_ratio_scores = fmet.get_likelihood_ratio_test_all(factor_scores)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
dip_scores, pval_scores = fmet.get_dip_test_all(factor_scores)
kurtosis_scores = fmet.get_factor_kurtosis_all(factor_scores)
outlier_sum_scores = fmet.get_outlier_sum_statistic_all(factor_scores)


### label dependent factor metrics
ASV_strain_all_arith = fmet.get_ASV_all(factor_scores, y_strain, mean_type='arithmetic')
ASV_cluster_all_arith = fmet.get_ASV_all(factor_scores, y_cluster, mean_type='arithmetic')
ASV_strain_all_geo = fmet.get_ASV_all(factor_scores, y_strain, mean_type='geometric')
ASV_cluster_all_geo = fmet.get_ASV_all(factor_scores, y_cluster, mean_type='geometric')


factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)

#### make a dictionary of all the metrics
all_metrics_dict = {'factor_entropy': factor_entropy_all,
                    'factor_variance': factor_variance_all, 
                    'ASV_strain_arith': ASV_strain_all_arith, 'ASV_strain_geo': ASV_strain_all_geo,
                    'ASV_cluster_arith': ASV_cluster_all_arith, 'ASV_cluster_geo': ASV_cluster_all_geo,
                    'factor_specificity': factor_specificity_all, 

                    'bic_km': bic_scores_km, 'calinski_harabasz_km': calinski_harabasz_scores_km,
                    'davies_bouldin_km': davies_bouldin_scores_km, 'silhouette_km': silhouette_scores_km,
                    'vrs_km': vrs_km, 'wvrs_km': wvrs_km,
                    'bic_gmm': bic_scores_gmm, 'silhouette_gmm': silhouette_scores_gmm,
                    'vrs_gmm': vrs_gmm, 'wvrs_gmm': wvrs_gmm,
                    'likelihood_ratio': likelihood_ratio_scores, 'bimodality_index': bimodality_index_scores,
                    'dip_score': dip_scores, 'dip_pval': pval_scores, 'kurtosis': kurtosis_scores,
                    'outlier_sum': outlier_sum_scores}

all_metrics_df = pd.DataFrame(all_metrics_dict)
factor_metrics = list(all_metrics_df.columns)
all_metrics_df.head()

###remove the likelihood_ratio metric - NA values??? TODO: check why
all_metrics_df = all_metrics_df.drop(columns=['likelihood_ratio'])

### visualize all_metrics_df 
all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)

fplot.plot_metric_correlation_clustermap(all_metrics_df)
fplot.plot_metric_dendrogram(all_metrics_df)

fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics)



fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=50)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=100)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=200)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=500)


