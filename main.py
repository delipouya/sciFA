import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt

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
pca_loading = pca.components_

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
pca_scores_rot, loadings_rot, rotmat = rot.get_varimax_rotated_scores(pca_scores, pca_loading)
print('loadings_rot shape: ', loadings_rot.shape)
print('rotmat shape: ', rotmat.shape)
print('pca_scores shape: ', pca_scores.shape)
print('pca_scores_rot shape: ', pca_scores_rot.shape)

factor_scores = pca_scores_rot
pca_loading = loadings_rot

'''
plot_pca(pca_scores_rot, pca,7, title='PCA of varimax rotated pearson residuals')
plot_umap(pca_scores_rot, pca, title='UMAP of the varimax-PCs on pearson residuals')


#### applying ICA to the gene expression data
num_components = 30
from sklearn.decomposition import FastICA
ica = FastICA(n_components=num_components)
ica_scores = ica.fit_transform(y)
ica_loading = ica.components_

pca_scores = ica_scores
pca_loading = ica_loading


#### applying ICA to the pearson residuals
num_components = 30
from sklearn.decomposition import FastICA
ica = FastICA(n_components=num_components)
ica_scores = ica.fit_transform(resid_pearson.T)
ica_loading = ica.components_

pca_scores = ica_scores
pca_loading = ica_loading


#### applying NMF to the gene expression data
num_components = 30

from sklearn.decomposition import NMF
model = NMF(n_components=num_components, init='random', random_state=0)
nmf_scores = model.fit_transform(y)
nmf_loading = model.components_

### check the shape of the data
print('shape of the data: ', y.shape)
print('shape of the pca scores: ', pca_scores.shape)
print('shape of the pca loading: ', pca_loading.shape)
print('shape of the nmf scores: ', nmf_scores.shape)
print('shape of the nmf loading: ', nmf_loading.shape)

pca_scores = nmf_scores
pca_loading = nmf_loading
'''


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
import functions_plotting as fplot
fplot.plot_fc_match_heatmap(mean_importance_df)
## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values


####################################
##### Factor metrics #####
####################################

#### AUC score
#### calculate the AUC of all the factors for all the covariate levels
AUC_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
AUC_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)
AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
fplot.plot_AUC_all_factors_df(AUC_all_factors_df, title='AUC of all the factors for all the covariate levels')

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

fplot.plot_scaled_variance_heatmap(SV_all_factors, all_covariate_levels, title='Scaled variance for all the factors')

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
fplot.plot_histogram(factor_scores[:,np.argmin(silhouette_scores_km)], title='factor with the minimum silhouette score')

### plot the correlation of the metrics
bimodality_metrics = ['bic', 'calinski_harabasz', 'davies_bouldin', 'silhouette', 'vrs', 'wvrs']
bimodality_scores = np.concatenate((bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km,
                                    silhouette_scores_km, vrs_km, wvrs_km), axis=0).reshape(len(bimodality_metrics), -1)
fplot.plot_metric_correlation(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))

fplot.plot_metric_heatmap(bimodality_scores, bimodality_metrics, title='Bimodality scores for all the factors')



### gmm clustering based bimodality metrics #### model-based clustering fmet.
bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores)
fplot.plot_metric_barplot(silhouette_scores_gmm, 'Silhouette score of each factor')
fplot.plot_metric_barplot(bic_scores_gmm, 'BIC score of each factor')

### plot the correlation of the metrics
bimodality_metrics = ['bic', 'silhouette', 'vrs', 'wvrs']
bimodality_scores = np.concatenate((bic_scores_gmm, silhouette_scores_gmm, 
                                    vrs_gmm, wvrs_gmm), axis=0).reshape(len(bimodality_metrics), -1)
fplot.plot_metric_correlation(pd.DataFrame(bimodality_scores.T, columns=bimodality_metrics))


### dip test based bimodality metrics
dip_scores, pval_scores = fmet.get_dip_test(factor_scores)
fplot.plot_histogram(dip_scores, 'Dip test score of each factor')
fplot.plot_histogram(pval_scores, 'Dip test p value of each factor')

fplot.plot_metric_barplot(dip_scores, 'Dip test score of each factor')

#### kurtosis: measure divergence from normal distribution
kurtosis_scores = fmet.get_kurtosis_scores(factor_scores)




####################################

#### label free factor metrics
factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
silhouette_score_all = fmet.get_kmeans_silhouette_scores(factor_scores)['silhouette']
factor_variance_all = fmet.get_factor_variance_all(factor_scores)


### label dependent factor metrics


ASV_protocol_all_arith = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='arithmetic')
ASV_cell_line_all_arith = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='arithmetic')
ASV_protocol_all_geo = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='geometric')
ASV_cell_line_all_geo = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='geometric')


factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)

#### make a dictionary of all the metrics
all_metrics_dict = {'factor_entropy': factor_entropy_all, 
                    'silhouette_score': silhouette_score_all,
                    'factor_variance': factor_variance_all, 
                    'ASV_protocol_arith': ASV_protocol_all_arith, 'ASV_protocol_geo': ASV_protocol_all_geo,
                    'ASV_cell_line_arith': ASV_cell_line_all_arith, 'ASV_cell_line_geo': ASV_cell_line_all_geo,
                    'factor_specificity': factor_specificity_all}

all_metrics_df = pd.DataFrame(all_metrics_dict)
factor_metrics = list(all_metrics_df.columns)
all_metrics_df.head()

fplot.plot_metric_correlation(all_metrics_df)

### scale the all_metrics matrix based on each metric 
all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)
fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_metric_heatmap_sb(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')