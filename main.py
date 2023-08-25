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


####################################
#### Running FA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_

plt.plot(pca.explained_variance_ratio_)
fplot.plot_pca_scMix(pca_scores, 2, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               covariate='protocol',
               title='PCA of the data matrix')

fplot.plot_pca_scMix(pca_scores, 2, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               covariate='cell_line',
               title='PCA of the data matrix')

fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['protocol'] , covariate='protocol')
fplot.plot_umap_scMix(pca_scores, colors_dict_scMix['cell_line'] , covariate='cell_line')


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
fplot.plot_fc_match_heatmap(mean_importance_df)


####################################
##### Factor metrics #####
####################################

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
all_covariate_levels = np.concatenate((y_protocol.unique(), y_cell_line.unique()), axis=0)

fplot.plot_scaled_variance_heatmap(SV_all_factors, all_covariate_levels, title='Scaled variance for all the factors')

ASV_protocol_all = fmet.get_ASV_all(factor_scores, y_protocol)
ASV_cell_line_all = fmet.get_ASV_all(factor_scores, y_cell_line)
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(ASV_protocol_all), colors_dict_scMix['protocol'], 
                          covariate='protocol', title='factor with maximum ASV of protocol')

#### Silhouette score
kmeans_silhouette_scores = fmet.get_kmeans_silhouette_scores(factor_scores)
silhouette_score_all = kmeans_silhouette_scores['silhouette']
kmeans_all = kmeans_silhouette_scores['kmeans']
fplot.plot_metric_barplot(silhouette_score_all, 'Silhouette score of each factor')
fplot.plot_factor_scatter(factor_scores, 0, np.argmax(silhouette_score_all), colors_dict_scMix['protocol'],
                            covariate='protocol', title='factor with maximum silhouette score')

fplot.plot_histogram(factor_scores[:,np.argmin(silhouette_score_all)], title='factor with the minimum silhouette score')

####################################


#### label free factor metrics
factor_entropy_all = fmet.get_factor_entropy_all(factor_scores)
silhouette_score_all = fmet.get_kmeans_silhouette_scores(factor_scores)['silhouette']
factor_variance_all = fmet.get_factor_variance_all(factor_scores)

### label dependent factor metrics
ASV_protocol_all = fmet.get_ASV_all(factor_scores, y_protocol)
ASV_cell_line_all = fmet.get_ASV_all(factor_scores, y_cell_line)
factor_specificity_all = fmet.get_all_factors_specificity(mean_importance_df)

#### make a dictionary of all the metrics
all_metrics_dict = {'factor_entropy': factor_entropy_all, 'silhouette_score': silhouette_score_all,
                    'factor_variance': factor_variance_all, 'ASV_protocol': ASV_protocol_all,
                    'ASV_cell_line': ASV_cell_line_all, 'factor_specificity': factor_specificity_all}

all_metrics_df = pd.DataFrame(all_metrics_dict)
factor_metrics = list(all_metrics_df.columns)
all_metrics_df.head()

fplot.plot_metric_correlation(all_metrics_df)

### scale the all_metrics matrix based on each metric 
all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)
fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_metric_heatmap_sb(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')