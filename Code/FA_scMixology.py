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



data_file_path = '/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad'
data = fproc.import_AnnData(data_file_path)
data, gene_idx = fproc.get_sub_data(data, random=False) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = fproc.get_data_array(data)
y_cell_line, y_sample, y_protocol = fproc.get_metadata_scMix(data)
data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()

colors_dict_scMix = fplot.get_colors_dict_scMix(y_protocol, y_cell_line)

plt_legend_cell_line = fplot.get_legend_patch(y_sample, colors_dict_scMix['cell_line'] )
plt_legend_protocol = fplot.get_legend_patch(y_sample, colors_dict_scMix['protocol'] )


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

num_pc = 5
### make a dictionary of colors for each sample in y_sample
fplot.plot_pca(pca_scores, num_pc, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title='PCA of pearson residuals - reg: lib size/protocol',
               plt_legend_list=plt_legend_cell_line)


fplot.plot_pca(pca_scores, num_pc, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title='PCA of pearson residuals - reg: lib size/protocol',
               plt_legend_list=plt_legend_protocol)



#### plot the loadings of the factors
fplot.plot_factor_loading(pca_loading.T, genes, 0, 2, fontsize=10, 
                    num_gene_labels=2,
                    title='Scatter plot of the loading vectors', 
                    label_x=False, label_y=False)

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

num_pc=29
fplot.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title='Varimax PCA of pearson residuals ',
               plt_legend_list=plt_legend_protocol)

fplot.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title='Varimax PCA of pearson residuals ',
               plt_legend_list=plt_legend_protocol)


######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
fplot.plot_pca(pca_scores_promax, 4, 
               cell_color_vec= colors_dict_scMix['protocol'], 
               legend_handles=True,
               title='Varimax PCA of pearson residuals ',
               plt_legend_list=plt_legend_protocol)

fplot.plot_pca(pca_scores_promax, 4, 
               cell_color_vec= colors_dict_scMix['cell_line'], 
               legend_handles=True,
               title='Varimax PCA of pearson residuals ',
               plt_legend_list=plt_legend_protocol)


#### plot the loadings of the factors
fplot.plot_factor_loading(varimax_loading, genes, 0, 4, fontsize=10, 
                    num_gene_labels=6,title='Scatter plot of the loading vectors', 
                    label_x=False, label_y=False)


####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'protocol'#'cell_line'
covariate_level = b'Dropseq'
covariate_vector = y_protocol

########################
factor_loading = pca_loading
factor_scores = pca_scores
########################


########################
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax



####################################
#### Mean Importance score
####################################

### calculate the mean importance of each covariate level
mean_importance_df_protocol = fmatch.get_mean_importance_all_levels(y_protocol, factor_scores, scale='standard', mean='arithmatic')
mean_importance_df_cell_line = fmatch.get_mean_importance_all_levels(y_cell_line, factor_scores, scale='standard', mean='arithmatic')

### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
mean_importance_df = pd.concat([mean_importance_df_protocol, mean_importance_df_cell_line], axis=0)
mean_importance_df.shape
fplot.plot_all_factors_levels_df(mean_importance_df, 
                                 title='F-C Match: Feature importance scores', 
                                 color='coolwarm',x_axis_fontsize=20, y_axis_fontsize=20, title_fontsize=22,
                               x_axis_tick_fontsize=32, y_axis_tick_fontsize=34)
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



### select the factors that are matched with any covariate level
matched_factor_index = np.where(matched_factor_dist>0)[0] 
### subset mean_importance_df to the matched factors
mean_importance_df_matched = mean_importance_df.iloc[:,matched_factor_index] 
## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched.columns.values

fplot.plot_all_factors_levels_df(mean_importance_df_matched, x_axis_label=x_labels_matched,
                                 title='F-C Match: Feature importance scores', color='coolwarm')

#### calculate the correlation of factors with library size
def get_factor_libsize_correlation(factor_scores, library_size):
    factor_libsize_correlation = np.zeros(factor_scores.shape[1])
    for i in range(factor_scores.shape[1]):
        factor_libsize_correlation[i] = np.corrcoef(factor_scores[:,i], library_size)[0,1]
    return factor_libsize_correlation

library_size = data.obs.nCount_originalexp
factor_libsize_correlation = get_factor_libsize_correlation(factor_scores, library_size)
### create a barplot of the factor_libsize_correlation

def plot_barplot(factor_libsize_correlation, x_labels=None, title=''):
    plt.figure(figsize=(10,5))
    if x_labels is None:
        x_labels = np.arange(factor_libsize_correlation.shape[0])
    plt.bar(x_labels, factor_libsize_correlation)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()

plot_barplot(factor_libsize_correlation, 
             title='Correlation of factors with library size')



####################################
#### evaluating bimodality score using simulated factors ####
####################################

#bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
#      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=False)
#bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores, time_eff=False)

silhouette_scores_km, vrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=True)
# silhouette_scores_gmm = fmet.get_gmm_scores(factor_scores, time_eff=True)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
bimodality_scores = bimodality_index_scores
### calculate the average between the silhouette_scores_km, vrs_km and bimodality_index_scores
bimodality_scores = np.mean([silhouette_scores_km, bimodality_index_scores], axis=0)


### label dependent factor metrics
ASV_geo_cell = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_line, mean_type='geometric')
ASV_geo_sample = fmet.get_ASV_all(factor_scores, y_sample, mean_type='geometric')


ASV_simpson_sample = fmet.get_all_factors_simpson(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_sample)))
ASV_simpson_cell = fmet.get_all_factors_simpson(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_cell_line)))


### calculate ASV based on entropy on the scaled variance per covariate for each factor
ASV_entropy_sample = fmet.get_factor_entropy_all(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_sample)))
ASV_entropy_cell = fmet.get_factor_entropy_all(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_cell_line)))

## calculate correlation between all ASV scores
ASV_list = [ASV_geo_cell, ASV_geo_sample, 
            ASV_simpson_sample, ASV_simpson_cell,
            ASV_entropy_sample, ASV_entropy_cell]
ASV_names = ['ASV_geo_cell',  'ASV_geo_sample', 
             'ASV_simpson_sample', 'ASV_simpson_cell', 
             'ASV_entropy_sample', 'ASV_entropy_cell']


### calculate the correlation between all ASV scores without a function
ASV_corr = np.zeros((len(ASV_list), len(ASV_list)))
for i in range(len(ASV_list)):
    for j in range(len(ASV_list)):
        ASV_corr[i,j] = np.corrcoef(ASV_list[i], ASV_list[j])[0,1]
ASV_corr_df = pd.DataFrame(ASV_corr)
### set the row and column names of ASV_corr_df
ASV_corr_df.index = ASV_names
ASV_corr_df.columns = ASV_names


### calculate diversity metrics
## simpson index: High scores (close to 1) indicate high diversity - meaning that the factor is not specific to any covariate level
## low simpson index (close to 0) indicate low diversity - meaning that the factor is specific to a covariate level
factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df) ## calculated for each factor in the importance matrix
factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

### calculate the average of the simpson index and entropy index
factor_simpson_entropy_meanimp = np.mean([factor_simpson_meanimp, factor_entropy_meanimp], axis=0)


#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)


####################################
##### Factor metrics #####
####################################

all_metrics_dict = {#'silhouette_km':silhouette_scores_km, 
                    #'vrs_km':vrs_km, #'silhouette_gmm':silhouette_scores_gmm, 
                    'bimodality_index':bimodality_index_scores,
                    'factor_variance':factor_variance_all, 

                    'homogeneity_cell':ASV_simpson_cell,
                    'homogeneity_sample':ASV_simpson_sample,

                    'homogeneity_cell_entropy':ASV_entropy_cell,
                    'homogeneity_sample_entropy':ASV_entropy_sample,
                
                    'factor_simpson_meanimp':[1-x for x in factor_simpson_meanimp], 
                    'factor_entropy_meanimp':[1-x for x in factor_entropy_meanimp]}


all_metrics_dict = {'bimodality':bimodality_scores, 
                    'specificity':[1-x for x in factor_simpson_entropy_meanimp],
                    'effect_size': factor_variance_all,
                    'homogeneity_cell':ASV_simpson_cell,
                    'homogeneity_sample':ASV_simpson_sample}

### check the length of all the metrics

all_metrics_df = pd.DataFrame(all_metrics_dict)
factor_metrics = list(all_metrics_df.columns)
all_metrics_df.head()

all_metrics_scaled = fmet.get_scaled_metrics(all_metrics_df)

fplot.plot_metric_correlation_clustermap(all_metrics_df)
fplot.plot_metric_dendrogram(all_metrics_df)
fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')


### subset all_merrics_scaled numpy array to only include the matched factors
all_metrics_scaled_matched = all_metrics_scaled[matched_factor_index,:]
fplot.plot_metric_heatmap(all_metrics_scaled_matched, factor_metrics, x_axis_label=x_labels_matched,
                          title='Scaled metrics for all the factors')

## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched.columns.values

