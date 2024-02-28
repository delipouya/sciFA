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

data_file_path = '/home/delaram/sciFA//Data/HumanLiverAtlas.h5ad'
data = fproc.import_AnnData(data_file_path)

data, gene_idx = fproc.get_sub_data(data, random=False) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = fproc.get_data_array(data)
y_sample, y_cell_type = fproc.get_metadata_humanLiver(data)
genes = data.var_names
### save genes as a csv file
pd.DataFrame(genes).to_csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv', index=False)

'''
### randomly subsample the cells to 2000 cells
sample_size = 2000
subsample_index = np.random.choice(y.shape[0], size=sample_size, replace=False)
y = y[subsample_index,:]
data = data[subsample_index,:]
y_sample = y_sample[subsample_index]
y_cell_type = y_cell_type[subsample_index]
'''
### get the indices of the highly variable genes

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='total_counts')

#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sample', data) 
x = np.column_stack((data.obs.total_counts, x_sample)) 
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

colors_dict_humanLiver = fplot.get_colors_dict_humanLiver(y_sample, y_cell_type)
plt_legend_sample = fplot.get_legend_patch(y_sample, colors_dict_humanLiver['sample'] )
plt_legend_cell_type = fplot.get_legend_patch(y_cell_type, colors_dict_humanLiver['cell_type'] )



plt.plot(pca.explained_variance_ratio_)

### make a dictionary of colors for each sample in y_sample
fplot.plot_pca(pca_scores, 4, 
               cell_color_vec= colors_dict_humanLiver['cell_type'], 
               legend_handles=True,
               title='PCA of gene expression data',
               plt_legend_list=plt_legend_cell_type)


fplot.plot_pca(pca_scores, 4, cell_color_vec= colors_dict_humanLiver['sample'])

#### plot the loadings of the factors
fplot.plot_factor_loading(pca_loading.T, genes, 0, 1, fontsize=10, 
                    num_gene_labels=2,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)

####################################
#### Matching between factors and covariates ######
####################################
covariate_name = 'cell_line'#''
covariate_level = 'nonInfMac'
#factor_scores = pca_scores
covariate_vector = y_cell_type


######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

num_pc = 20
fplot.plot_pca(pca_scores_varimax, num_pc, cell_color_vec= colors_dict_humanLiver['sample'],
               legend_handles=True,
               title='varimax-PCA of pearson residual',
               plt_legend_list=plt_legend_cell_type)
fplot.plot_pca(pca_scores_varimax, num_pc, cell_color_vec= colors_dict_humanLiver['cell_type'],
               legend_handles=True,
               title='varimax-PCA of pearson residual',
               plt_legend_list=plt_legend_cell_type)





fplot.plot_factor_loading(varimax_loading, genes, 0, 1, fontsize=10, 
                    num_gene_labels=2,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)




################################################
#### make the loading scatter plot with histograms on the sides
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, color='black', s=2, alpha=0.5)

    # now determine nice limits by hand:
    binwidth = 0.009
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

x = varimax_loading[:,0]
y = varimax_loading[:,1]
# Start with a square Figure.
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
scatter_hist(x, y, ax, ax_histx, ax_histy)
####################################################################



######## Applying promax rotation to the factor scores
rotation_results_promax = rot.promax_rotation(pca_loading.T)
promax_loading = rotation_results_promax['rotloading']
pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_humanLiver['strain'])
fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_humanLiver['cluster'])


########################
factor_loading = pca_loading
factor_scores = pca_scores

########################

########################
factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax





### find unique covariate levels
y_cell_type_unique = np.unique(y_cell_type)

####################################
#### Mean Importance score
####################################

### calculate the mean importance of each covariate level
mean_importance_df_sample = fmatch.get_mean_importance_all_levels(y_sample, factor_scores,scale='standard', mean='arithmatic',time_eff=True)
mean_importance_df_cell_type = fmatch.get_mean_importance_all_levels(y_cell_type, factor_scores,scale='standard', mean='arithmatic',time_eff=True)
# 
### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
mean_importance_df = pd.concat([mean_importance_df_sample, mean_importance_df_cell_type], axis=0)
mean_importance_df.shape

mean_importance_df = mean_importance_df_cell_type

fplot.plot_all_factors_levels_df(mean_importance_df,
                                 color='coolwarm', title='',
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                 x_axis_tick_fontsize=36, y_axis_tick_fontsize=40, 
                                 save=True, save_path='../Plots/mean_importance_df_matched_human_liver.pdf')
## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values


######################## run umap on factor scores
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(factor_scores)


plt.figure()
plt.rcParams['axes.facecolor'] = 'white'
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors_dict_humanLiver['cell_type'], s=1)
### locate the legend outside of the plot
plt.legend(handles=plt_legend_cell_type, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) 


#### project all factor_score onto the first two umap components in a loop
plt.figure()
plt.rcParams['axes.facecolor'] = 'white'
for i in range(factor_scores.shape[1]):
    plt.scatter(embedding[:, 0], embedding[:, 1], c=factor_scores[:,i], s=2, cmap='Spectral')
    plt.colorbar(boundaries=np.arange(10)-0.5).set_ticks(np.arange(9))
    plt.title('factor {}'.format(i))
    plt.show()
    plt.close()





##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level
### plot the histogram of all the values in mean importance scores
### choosing a threshold for the feature importance scores
threshold = fmatch.get_otsu_threshold(mean_importance_df.values.flatten())

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance scores',
                        title='F-C Match: Feature importance scores', threshold=threshold)

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance score',
                        title='Feature importance distribution', threshold=threshold,
                        save=True, save_path='../Plots/histogram_imp_humanliver.pdf',
                        xlabel_fontsize=21, ylabel_fontsize=20, title_fontsize=22,
                   xticks_fontsize=18,yticks_fontsize=16)

matched_factor_dist, percent_matched_fact = fmatch.get_percent_matched_factors(mean_importance_df, threshold)
matched_covariate_dist, percent_matched_cov = fmatch.get_percent_matched_covariate(mean_importance_df, threshold=threshold)

print('percent_matched_fact: ', percent_matched_fact)
print('percent_matched_cov: ', percent_matched_cov)
fplot.plot_matched_factor_dist(matched_factor_dist, save=True, 
                               save_path='../Plots/num_matched_factor_humanliver.pdf')
fplot.plot_matched_covariate_dist(matched_covariate_dist, covariate_levels=all_covariate_levels,
                                  save=True, save_path='../Plots/num_matched_cov_humanliver.pdf')

#### print the factors that are not matched with any covariate level + 1
print('factors that are not matched with any covariate level: ', np.where(matched_factor_dist==0)[0])
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

library_size = data.obs.total_counts
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


#### concatenate the factor scores with the metadata and umap embedding
factor_scores_df = pd.DataFrame(factor_scores)
print(factor_scores_df.shape)
factor_scores_df.columns = ['factor_{}'.format(i) for i in range(factor_scores.shape[1])]
factor_scores_df['SAMPLE'] = y_sample
factor_scores_df['CELL_TYPE'] = y_cell_type
factor_scores_df['umap_1'] = embedding[:,0]
factor_scores_df['umap_2'] = embedding[:,1]
print(factor_scores_df.shape)

### add add the columns in the data.obs to the factor_scores_df as new columns
for col in data.obs.columns:
    factor_scores_df[col] = data.obs[col].values
print(factor_scores_df.shape)
print(factor_scores_df.head())

### add rownames of data.obs to the factor_scores_df
factor_scores_df['id'] = data.obs.index.values
#### save the factor_scores_df as a csv file
factor_scores_df.to_csv('/home/delaram/sciFA/Results/factor_scores_umap_df_humanlivermap.csv', index=False)
### save loadings as a csv file
pd.DataFrame(factor_loading).to_csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv', index=False)


####################################
#### evaluating bimodality score using simulated factors ####
####################################

#bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
#      vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=False)
#bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores, time_eff=False)

silhouette_scores_km, vrs_km = fmet.get_kmeans_scores(factor_scores, time_eff=True)
# silhouette_scores_gmm = fmet.get_gmm_scores(factor_scores, time_eff=True)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
### calculate the average between the silhouette_scores_km, vrs_km and bimodality_index_scores
bimodality_scores = np.mean([silhouette_scores_km, bimodality_index_scores], axis=0)


### label dependent factor metrics
ASV_geo_cell = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_type, mean_type='geometric')
ASV_geo_sample = fmet.get_ASV_all(factor_scores, y_sample, mean_type='geometric')


ASV_simpson_sample = fmet.get_all_factors_simpson(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_sample)))
ASV_simpson_cell = fmet.get_all_factors_simpson(pd.DataFrame(fmet.get_factors_SV_all_levels(factor_scores, y_cell_type)))

## calculate correlation between all ASV scores
ASV_list = [ASV_geo_cell, ASV_geo_sample, ASV_simpson_sample, ASV_simpson_cell]
ASV_names = ['ASV_geo_cell', 'ASV_geo_sample', 'ASV_simpson_sample', 'ASV_simpson_cell' ]
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

all_metrics_dict = {'silhouette_km':silhouette_scores_km, 
                    'vrs_km':vrs_km, #'silhouette_gmm':silhouette_scores_gmm, 
                    'bimodality_index':bimodality_index_scores,
                    'factor_variance':factor_variance_all, 

                    'homogeneity_cell':ASV_simpson_cell,
                    'homogeneity_sample':ASV_simpson_sample,
                
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





fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=50)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=100)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=200)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=500)


