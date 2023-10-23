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

data = fproc.import_AnnData('./Data/scMix_3cl_merged.h5ad')
y_cell_line, y_sample, y_protocol = fproc.get_metadata_scMix(data)

data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()
y, num_cells, num_genes = fproc.get_data_array(data)
y = fproc.get_sub_data(y)
colors_dict_scMix = fplot.get_colors_dict_scMix(y_protocol, y_cell_line)
genes = data.var_names


####################################
#### Running PCA on the data ######
####################################
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape #(factors, genes)
plt.plot(pca.explained_variance_ratio_)
fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title='PCA of gene expression data', legend_handles=True,
               plt_legend_dict=fplot.plt_legend_dict)

fplot.plot_pca(pca_scores, 6, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title='PCA of gene expression data', legend_handles=True,
               plt_legend_dict=fplot.plt_legend_dict)


#### plot the loadings of the factors
fplot.plot_factor_loading(pca_loading.T, genes, 0, 2, fontsize=10, 
                    num_gene_labels=2,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)

####################################
#### fit GLM to each gene ######
####################################

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_originalexp')

#### design matrix - library size and sample
x_protocol = fproc.get_design_mat('protocol', data) 
x = np.column_stack((data.obs.nCount_originalexp, x_protocol)) 
x = sm.add_constant(x) ## adding the intercept

glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)

################################################
#### Running PCA on the pearson residual ######
################################################

### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape
plt.plot(pca.explained_variance_ratio_)
fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title='PCA of pearson residuals - reg: lib size/protocol',
               legend_handles=True,
               plt_legend_dict=fplot.plt_legend_dict)

fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title='PCA of pearson residuals - reg: lib size/protocol',
               legend_handles=True,
               plt_legend_dict=fplot.plt_legend_dict)

fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['protocol'] , covariate='protocol', title='UMAP')
fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['cell_line'] , covariate='cell_line',title='UMAP')


###################################################
#### Matching between factors and covariates ######
###################################################
covariate_name = 'cell_line'#'cell_line'
#factor_scores = pca_scores
covariate_vector = y_cell_line
y_cell_line.unique()
covariate_level = 'HCC827'


######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])
fplot.plot_pca(pca_scores_varimax, 30, cell_color_vec= colors_dict_scMix['protocol'],
               title='Varimax PCA of pearson residuals ',
               legend_handles=True,plt_legend_dict=fplot.plt_legend_dict, covariate='protocol')
fplot.plot_pca(pca_scores_varimax, 30, cell_color_vec= colors_dict_scMix['cell_line'],
               title='Varimax PCA of pearson residuals',
               legend_handles=True,plt_legend_dict=fplot.plt_legend_dict, covariate='cell_line')


######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_scMix['protocol'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_scMix['cell_line'])



#### plot the loadings of the factors
fplot.plot_factor_loading(varimax_loading, genes, 0, 2, fontsize=10, 
                    num_gene_labels=6,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)


####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'protocol'#'cell_line'
covariate_level = b'Dropseq'
covariate_vector = y_protocol

########################
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax
########################
factor_loading = pca_loading
factor_scores = pca_scores
########################


####################################
#### Mean Importance score
####################################

### calculate the mean importance of each covariate level
mean_importance_df_protocol = fmatch.get_mean_importance_all_levels(y_protocol, factor_scores)
mean_importance_df_cell_line = fmatch.get_mean_importance_all_levels(y_cell_line, factor_scores)

### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
mean_importance_df = pd.concat([mean_importance_df_protocol, mean_importance_df_cell_line], axis=0)
mean_importance_df.shape
fplot.plot_all_factors_levels_df(mean_importance_df, title='F-C Match: Feature importance scores', color='coolwarm')
## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values

##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level
### plot the histogram of all the values in mean importance scores
fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                     title='F-C Match: Feature importance scores') 

### choosing a threshold for the feature importance scores

threshold = 0.3
threshold = fmatch.get_otsu_threshold(mean_importance_df.values.flatten())

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                        title='F-C Match: Feature importance scores', threshold=threshold)

matched_factor_dist, percent_matched_fact = fmatch.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched_cov = fmatch.get_percent_matched_covariate(mean_importance_df, threshold=threshold)

print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
fplot.plot_matched_factor_dist(matched_factor_dist)
fplot.plot_matched_covariate_dist(matched_covariate_dist, covariate_levels=all_covariate_levels)

####################################
#### AUC score
####################################
#### calculate the AUC of all the factors for all the covariate levels
AUC_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
AUC_all_factors_df_cell_line, wilcoxon_pvalue_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)

AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
wilcoxon_pvalue_all_factors_df = pd.concat([wilcoxon_pvalue_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_cell_line], axis=0)

fplot.plot_all_factors_levels_df(AUC_all_factors_df, 
                                 title='F-C Match: AUC scores', color='coolwarm') #'YlOrBr'

### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
## list of color maps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

threshold = 0.3
threshold = fmatch.get_otsu_threshold(AUC_all_factors_df.values.flatten())

fplot.plot_histogram(AUC_all_factors_df.values.flatten(), xlabel='AUC scores',
                        title='F-C Match: AUC score', threshold=threshold)

matched_factor_dist, percent_matched_fact = fmatch.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched_cov = fmatch.get_percent_matched_covariate(mean_importance_df, threshold=threshold)

print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
fplot.plot_matched_factor_dist(matched_factor_dist)
fplot.plot_matched_covariate_dist(matched_covariate_dist, covariate_levels=all_covariate_levels)


####################################
##### Factor metrics #####
####################################

AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
fplot.plot_all_factors_levels_df(AUC_all_factors_df_1,
                                    title='Homogeneity: 1-AUC scores', color='viridis') #

### p value score is not a good indicative of the homogeneity of the factors
#fplot.plot_all_factors_levels_df(wilcoxon_pvalue_all_factors_df, 
#                                 title='Homogeneity: Wilcoxon pvalue', color='rocket_r')

#### Specificity 
factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)
fplot.plot_metric_barplot(factor_specificity_all, 'Specificity of each factor')
#highest_vote_factor = a_cov_match_factor_dict[covariate_level]; i = int(highest_vote_factor[2:]) - 1
factor_i = 7
fplot.plot_factor_scatter(factor_scores, 0, factor_i, colors_dict_scMix['protocol'], covariate='protocol')

#### Entropy 
factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
fplot.plot_metric_barplot(factor_entropy_all, 'Entropy of each factor')
fplot.plot_histogram(factor_scores[:,np.argmax(factor_entropy_all)], title='factor with the maximum entropy')

#### Variance
factor_variance_all = fmet.get_factor_variance_all(factor_scores)
fplot.plot_metric_barplot(factor_variance_all, 'Variance of each factor')
fplot.plot_factor_scatter(factor_scores, 0, np.argmin(factor_variance_all), colors_dict_scMix['protocol'], 
                          covariate='protocol', title='factor with the minimum variance')

#### Scaled variance
SV_all_factors_protocol = fmet.get_factors_SV_all_levels(factor_scores, y_protocol)
SV_all_factors_cell_line = fmet.get_factors_SV_all_levels(factor_scores, y_cell_line)
SV_all_factors = np.concatenate((SV_all_factors_protocol, SV_all_factors_cell_line), axis=0)
#all_covariate_levels = np.concatenate((y_protocol.unique(), y_cell_line.unique()), axis=0)

### convert to SV_all_factors to dataframe
SV_all_factors_df = pd.DataFrame(SV_all_factors)
SV_all_factors_df.columns = AUC_all_factors_df.columns
SV_all_factors_df.index = AUC_all_factors_df.index
## scale each factor from 0 to 1
SV_all_factors_df = SV_all_factors_df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

fplot.plot_all_factors_levels_df(SV_all_factors_df,
                                    title='Homogeneity: scaled variance scores', color='RdPu')
fplot.plot_all_factors_levels(SV_all_factors, all_covariate_levels, 
                              title='Scaled variance for all the factors', color='RdPu')


fplot.plot_all_factors_levels(SV_all_factors, all_covariate_levels, 
                              title='Scaled variance for all the factors', color='RdPu')


ASV_protocol_all = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='arithmetic')
ASV_cell_line_all = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='arithmetic')
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(ASV_protocol_all), colors_dict_scMix['protocol'], 
                          covariate='protocol', title='factor with maximum ASV of protocol')



#### bimodality scores

### kmeans clustering based bimodality metrics
### TODO: try initializing kmeans with the min and max factor scores for cluster centers
bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores)

fplot.plot_metric_barplot(silhouette_scores_km, 'Silhouette score of each factor')
fplot.plot_metric_barplot(calinski_harabasz_scores_km, 'Calinski Harabasz score of each factor')
fplot.plot_metric_barplot(wvrs_km, 'weighted variance ratio score of each factor')
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(silhouette_scores_km), colors_dict_scMix['protocol'],
                            covariate='protocol', title='factor with maximum silhouette score')
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
ASV_protocol_all_arith = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='arithmetic')
ASV_cell_line_all_arith = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='arithmetic')
ASV_protocol_all_geo = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='geometric')
ASV_cell_line_all_geo = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='geometric')

### create a dictionaty annd thenn a dataframe of all the ASV metrics arrays
ASV_all_factors_dict = {'ASV_protocol_arith': ASV_protocol_all_arith, 'ASV_protocol_geo': ASV_protocol_all_geo,
                        'ASV_cell_line_arith': ASV_cell_line_all_arith, 'ASV_cell_line_geo': ASV_cell_line_all_geo}
ASV_all_factors_df = pd.DataFrame(ASV_all_factors_dict)
### visualize the table with numbers using sns
import seaborn as sns
sns.set(font_scale=0.7)
### change teh figure size
plt.figure(figsize=(5, 10))
sns.heatmap(ASV_all_factors_df, cmap='coolwarm', annot=True, fmt='.2f', cbar=False)
plt.xlabel('Covariate levels')
plt.ylabel('Factor number')
plt.title('ASV scores for all the factors')
plt.show()

#### Scaled variance
SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, covariate_vector)


### calculate specificity
factor_specificity_meanimp = fmet.get_all_factors_specificity(mean_importance_df)
factor_specificity_AUC = fmet.get_all_factors_specificity(AUC_all_factors_df)


#### make a dictionary of all the metrics
all_metrics_dict = {'factor_entropy': factor_entropy_all,
                    'factor_variance': factor_variance_all, 
                    'ASV_protocol_arith': ASV_protocol_all_arith, 'ASV_protocol_geo': ASV_protocol_all_geo,
                    'ASV_cell_line_arith': ASV_cell_line_all_arith, 'ASV_cell_line_geo': ASV_cell_line_all_geo,
                    'factor_specificity_meanimp': factor_specificity_meanimp, 'factor_specificity_AUC':factor_specificity_AUC,

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

all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)

fplot.plot_metric_correlation_clustermap(all_metrics_df)
fplot.plot_metric_dendrogram(all_metrics_df)

fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics)

### subset the all_metrics_df to the metrics_to_keep
metrics_to_keep = ['vrs_km', 'silhouette_gmm', 'bimodality_index', 
 'factor_variance', 'factor_specificity_meanimp', 'ASV_protocol_geo', 'ASV_cell_line_geo']

all_metrics_df_sub = all_metrics_df[metrics_to_keep]
factor_metrics_sub = list(all_metrics_df_sub.columns)
all_metrics_scaled_sub = fmet.get_scaled_metrics(all_metrics_df_sub)
fplot.plot_metric_heatmap(all_metrics_scaled_sub, factor_metrics_sub, title='Scaled metrics for all the factors')


fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=50)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=100)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=200)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=500)


