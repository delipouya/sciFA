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


data_file_path = '/home/delaram/sciFA//Data/inputdata_rat_set1_countData_2.h5ad'
data = fproc.import_AnnData(data_file_path)
data, gene_idx = fproc.get_sub_data(data, num_genes=10000,random=False) # subset the data to num_genes HVGs
y, genes, num_clusters, num_genes = fproc.get_data_array(data)

### add the cluster annotations to teh metadata
annotation_data_file = '/home/delaram/sciFA//Data/TLH_annotation.csv'
annotation_data = pd.read_csv(annotation_data_file, index_col=0)
annotation_data['cluster_value'] = annotation_data.index.values
annotation_data['cluster_value']=annotation_data['cluster_value'].astype(int)

### convert meta_data to pandas dataframe
meta_data = pd.DataFrame(data.obs)
meta_data['cluster']=meta_data['cluster'].astype(int)

### check if meta_data['cluster'] and annotation_data['cluster_value'] have the same data type
print('meta_data[cluster] data type: ', meta_data['cluster'].dtype)
print('annotation_data[cluster_value] data type: ', annotation_data['cluster_value'].dtype)

### add the cluster annotations to the metadata 
meta_data = pd.merge(meta_data, annotation_data, how='left', left_on='cluster', right_on='cluster_value')
data.obs = meta_data
y_sample, y_strain, y_cluster, y_cell_type = fproc.get_metadata_ratLiver(data)

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_RNA')
'''
#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sample', data) 
x = np.column_stack((data.obs.nCount_RNA, x_sample)) 
x = sm.add_constant(x) ## adding the intercept
'''
#### calculate row and column sum of y
row_sum_y = np.sum(y, axis=1)
col_sum_y = np.sum(y, axis=0)

### takes 6 mins to fit 10000 genes
### fit GLM to each gene
glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_clusters)
print('y shape: ', y.shape) # (num_clusters, num_genes)
y = resid_pearson.T # (num_clusters, num_genes)
print('y shape: ', y.shape) # (num_clusters, num_genes)

####################################
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape #(factors, genes)

colors_dict_ratLiver = fplot.get_colors_dict_ratLiver(y_sample, y_strain, y_cell_type)

plt_legend_sample = fplot.get_legend_patch(y_sample, colors_dict_ratLiver['sample'] )
plt_legend_cell_type = fplot.get_legend_patch(y_cell_type, colors_dict_ratLiver['cell_type'] )
plt_legend_strain = fplot.get_legend_patch(y_strain, colors_dict_ratLiver['strain'] )
plt.plot(pca.explained_variance_ratio_)


### make a dictionary of colors for each sample in y_sample
fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_ratLiver['cell_type'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_cell_type)

fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_ratLiver['sample'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_sample)


fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_ratLiver['strain'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_strain)

####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'strain'#'cluster_line'
covariate_level = 'DA'
#factor_scores = pca_scores
covariate_vector = y_strain

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])
num_pc = 30
fplot.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_ratLiver['cell_type'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_cell_type)
fplot.plot_pca(pca_scores_varimax, num_pc, 
               cell_color_vec= colors_dict_ratLiver['strain'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_strain)



### convert the varimax_loading to a dataframe
varimax_loading_df = pd.DataFrame(varimax_loading)
### name columns F1 to F30
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
varimax_loading_df.index = genes

### concatenate the pca_scores_varimax with the data.obs dataframe in a separate dataframe
pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]
pca_scores_varimax_df.index = data.obs.index.values
pca_scores_varimax_df_merged = pd.concat([data.obs, pca_scores_varimax_df], axis=1)


### save the pca_scores_varimax_df_merged to a csv file
pca_scores_varimax_df_merged.to_csv('../Results/pca_scores_varimax_df_merged_ratLiver.csv')
## save the varimax_loading_df and varimax_scores to a csv file
varimax_loading_df.to_csv('../Results/varimax_loading_df_ratLiver.csv')






######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
fplot.plot_pca(pca_scores_promax, 9, cluster_color_vec= colors_dict_ratLiver['strain'])
fplot.plot_pca(pca_scores_promax, 9, cluster_color_vec= colors_dict_ratLiver['cell_type'])

########################



####################################
#### Mean Importance score
####################################
factor_loading = varimax_loading #pca_loading
factor_scores = pca_scores_varimax #pca_scores_promax #pca_scores_varimax

### calculate the mean importance of each covariate level
mean_importance_df_sample = fmatch.get_mean_importance_all_levels(y_sample, factor_scores,scale='standard', 
                                                                  mean='arithmatic',time_eff=True)
mean_importance_df_strain = fmatch.get_mean_importance_all_levels(y_strain, factor_scores,scale='standard', 
                                                                  mean='arithmatic',time_eff=True)
mean_importance_df_cell_type = fmatch.get_mean_importance_all_levels(y_cell_type, factor_scores,scale='standard',
                                                                   mean='arithmatic',time_eff=True)
# 
### concatenate mean_importance_df_protocol and mean_importance_df_cluster_line
mean_importance_df = pd.concat([ mean_importance_df_strain, mean_importance_df_cell_type], axis=0) #mean_importance_df_sample,
fplot.plot_all_factors_levels_df(mean_importance_df, title='F-C Match: Feature importance scores', 
                                 color='coolwarm')


## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values

##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level
### plot the histogram of all the values in mean importance scores
### choosing a threshold for the feature importance scores
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
## add index 19 to the matched_factor_index
matched_factor_index = np.append(matched_factor_index, 19)
### subset mean_importance_df to the matched factors
mean_importance_df_matched_sub = mean_importance_df.iloc[:,matched_factor_index] 
## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched_sub.columns.values

fplot.plot_all_factors_levels_df(mean_importance_df_matched_sub, x_axis_label=x_labels_matched,
                                   color='coolwarm', title='',
                                 x_axis_fontsize=45, y_axis_fontsize=44, title_fontsize=40,
                                 x_axis_tick_fontsize=39, y_axis_tick_fontsize=41, 
                                 legend_fontsize=38,
                                 save=True, save_path='../Plots/mean_importance_df_matched_ratliver.pdf')




a_cov_level = 'DA'#all_covariate_levels[0]
### create sorted bar plot of factors with high scores for a_cov_level based on mean_importance_df

x_labels = None
if x_labels is None:
      x_labels = mean_importance_df.columns.values
plt.figure(figsize=(12,5))
a_cov_level_score = mean_importance_df.loc[a_cov_level,:]
### sort x_labels and a_cov_level_score based on a_cov_level_score
a_cov_level_score_sorted, x_labels_sorted = zip(*sorted(zip(a_cov_level_score, x_labels), reverse=True))
plt.bar(x_labels_sorted, a_cov_level_score_sorted)
### add title 
plt.title('Sorted factor feature importance scores for Stimulated covariate', fontsize=22)
### decrease title font size

plt.xticks(rotation=90, fontsize=22)
### increase y axis ticks size
plt.yticks(fontsize=25)
#plt.savefig('../Plots/sorted_factor_feature_importance_scores_'+a_cov_level+'.pdf')
plt.show()
### save the plot


#### calculate the correlation of factors with library size
def get_factor_libsize_correlation(factor_scores, library_size):
    factor_libsize_correlation = np.zeros(factor_scores.shape[1])
    for i in range(factor_scores.shape[1]):
        factor_libsize_correlation[i] = np.corrcoef(factor_scores[:,i], library_size)[0,1]
    return factor_libsize_correlation

library_size = data.obs.nCount_RNA
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

##########################################
############## saving output to csv files
##########################################
### convert the varimax_loading to a dataframe
varimax_loading_df = pd.DataFrame(varimax_loading)
### name columns F1 to F30
varimax_loading_df.columns = ['F'+str(i) for i in range(1, varimax_loading_df.shape[1]+1)]
varimax_loading_df.index = genes

### concatenate the pca_scores_varimax with the data.obs dataframe in a separate dataframe
pca_scores_varimax_df = pd.DataFrame(pca_scores_varimax)
pca_scores_varimax_df.columns = ['F'+str(i) for i in range(1, pca_scores_varimax_df.shape[1]+1)]
pca_scores_varimax_df.index = data.obs.index.values
pca_scores_varimax_df_merged = pd.concat([data.obs, pca_scores_varimax_df], axis=1)
### save the pca_scores_varimax_df_merged to a csv file
pca_scores_varimax_df_merged.to_csv('../Results/pca_scores_varimax_df_ratliver_libReg_'+
                                    str(num_genes)+'.csv')
## save the varimax_loading_df and varimax_scores to a csv file
varimax_loading_df.to_csv('../Results/varimax_loading_df_ratliver_libReg_'+
                          str(num_genes)+'.csv')

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

#bimodality_scores = np.mean([silhouette_scores_km, bimodality_index_scores], axis=0)

#### Scaled variance
### calculate the number of levels in the covariate vectors cell_type sample and sex
num_levels_cell_type = len(np.unique(y_cell_type))
num_levels_sample = len(np.unique(y_sample))
num_levels_strain = len(np.unique(y_strain))
print('num_levels_cell_type: ', num_levels_cell_type)
print('num_levels_sample: ', num_levels_sample)
print('num_levels_strain: ', num_levels_strain)

#SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, y_cell_type) 
### label dependent factor metrics
#ASV_all_arith = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')

ASV_all_arith_stim = fmet.get_ASV_all(factor_scores, y_strain, mean_type='arithmetic')

#SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, y_cell_type) 
### label dependent factor metrics
#ASV_all_arith = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')

### calculate diversity metrics
## simpson index: High scores (close to 1) indicate high diversity - meaning that the factor is not specific to any covariate level
## low simpson index (close to 0) indicate low diversity - meaning that the factor is specific to a covariate level
#factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
#factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df_cell_type) ## calculated for each factor in the importance matrix
#factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df) ## calculated for each factor in the importance matrix
factor_simpson_meanimp_cell_type = fmet.get_all_factors_simpson(mean_importance_df_cell_type) 
factor_simpson_meanimp_sample = fmet.get_all_factors_simpson(mean_importance_df_sample) 

#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)

### calculate diversity metrics
## simpson index: High scores (close to 1) indicate high diversity - meaning that the factor is not specific to any covariate level
## low simpson index (close to 0) indicate low diversity - meaning that the factor is specific to a covariate level
factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

### calculate the average of the simpson index and entropy index
factor_simpson_entropy_meanimp = np.mean([factor_simpson_meanimp, factor_entropy_meanimp], axis=0)



SV_strain_df = pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_strain))
### plot the heatmap of SV_strain_df
plt.figure(figsize=(10,10))
plt.imshow(SV_strain_df, cmap='coolwarm')
plt.xticks(np.arange(SV_strain_df.shape[1]), SV_strain_df.columns.values, rotation=90)
plt.yticks(np.arange(SV_strain_df.shape[0]), SV_strain_df.index.values)
plt.colorbar()
plt.show()


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


                    'homogeneity_cluster_simpson':ASV_simpson_cluster,
                    'homogeneity_cluster_entropy':ASV_entropy_cluster,
                    'homogeneity_cluster_geo':ASV_geo_cluster,


                    'homogeneity_strain_simpson': ASV_simpson_strain,
                    'homogeneity_strain_entropy': ASV_entropy_strain,
                    'homogeneity_strain_geo': ASV_geo_strain,

                    'homogeneity_sample_simpson':ASV_simpson_sample,
                    'homogeneity_sample_entropy':ASV_entropy_sample,
                    'homogeneity_sample_geo':ASV_geo_sample,
                
                    'factor_simpson_meanimp':[1-x for x in factor_simpson_meanimp], 
                    'factor_entropy_meanimp':[1-x for x in factor_entropy_meanimp]}


all_metrics_dict = {'silhouette_km':silhouette_scores_km, 
                    'vrs_km':vrs_km, #'silhouette_gmm':silhouette_scores_gmm, 
                    'bimodality_index':bimodality_index_scores,
                    'factor_variance':factor_variance_all, 

                    'homogeneity_cluster':ASV_simpson_cluster,
                    'homogeneity_sample':ASV_simpson_sample,
                
                    'factor_simpson_meanimp':[1-x for x in factor_simpson_meanimp], 
                    'factor_entropy_meanimp':[1-x for x in factor_entropy_meanimp]}


all_metrics_dict = {'bimodality':bimodality_scores, 
                    'specificity':[1-x for x in factor_simpson_entropy_meanimp],
                    'effect_size': factor_variance_all,
                    'homogeneity_cluster':ASV_simpson_cluster,
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


