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
import rotations as rot
import constants as const

np.random.seed(10)

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
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
factor_loading = pca.components_

plt.plot(pca.explained_variance_ratio_)
fplot.plot_pca_scMix(pca_scores, 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title='PCA of the data matrix')

fplot.plot_pca_scMix(pca_scores, 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title='PCA of the data matrix')

fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['protocol'] , covariate='protocol')
fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['cell_line'] , covariate='cell_line')


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
      
      
####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'protocol'#'cell_line'
covariate_level = b'Dropseq'
factor_scores = pca_scores
covariate_vector = y_protocol

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
threshold = 0.3

matched_factor_dist, percent_matched = fmatch.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched = fmatch.get_percent_matched_covariate(mean_importance_df, threshold)

fplot.plot_matched_factor_dist(matched_factor_dist)
fplot.plot_matched_covariate_dist(matched_covariate_dist)


####################################
##### Factor metrics #####
####################################
#### AUC score
#### calculate the AUC of all the factors for all the covariate levels
AUC_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
AUC_all_factors_df_cell_line, wilcoxon_pvalue_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)

AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
wilcoxon_pvalue_all_factors_df = pd.concat([wilcoxon_pvalue_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_cell_line], axis=0)

fplot.plot_all_factors_levels_df(AUC_all_factors_df, 
                                 title='F-C Match: AUC scores', color='YlOrBr')

### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
## list of color maps: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html


AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
fplot.plot_all_factors_levels_df(AUC_all_factors_df_1,
                                    title='Homogeneity: 1-AUC scores', color='hot')

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


factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)

#### make a dictionary of all the metrics
all_metrics_dict = {'factor_entropy': factor_entropy_all,
                    'factor_variance': factor_variance_all, 
                    'ASV_protocol_arith': ASV_protocol_all_arith, 'ASV_protocol_geo': ASV_protocol_all_geo,
                    'ASV_cell_line_arith': ASV_cell_line_all_arith, 'ASV_cell_line_geo': ASV_cell_line_all_geo,
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

all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)

fplot.plot_metric_correlation_clustermap(all_metrics_df)
fplot.plot_metric_dendrogram(all_metrics_df)

fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics)



fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=50)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=100)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=200)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=500)


