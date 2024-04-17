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


data_file_path = '/home/delaram/sciFA//Data/Human_Kidney_data.h5ad'
data = fproc.import_AnnData(data_file_path)

data, gene_idx = fproc.get_sub_data(data, random=False) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = fproc.get_data_array(data)
y_sample, y_sex, y_cell_type, y_cell_type_sub = fproc.get_metadata_humanKidney(data)


####################################
#### Running PCA on the data ######
####################################
### using pipeline to scale the gene expression data first
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_ 
pca_loading.shape #(factors, genes)

colors_dict_humanKidney = fplot.get_colors_dict_humanKidney(y_sample, y_sex, y_cell_type)

plt_legend_celltype = fplot.get_legend_patch(y_cell_type, colors_dict_humanKidney['cell_type'] )
plt_legend_sex = fplot.get_legend_patch(y_sex, colors_dict_humanKidney['sex'] )
plt_legend_sample = fplot.get_legend_patch(y_sample, colors_dict_humanKidney['sample'] )

plt.plot(pca.explained_variance_ratio_)

### make a dictionary of colors for each sample in y_sample
fplot.plot_pca(pca_scores, 3, cell_color_vec= colors_dict_humanKidney['cell_type'], 
               legend_handles=True,plt_legend_list=plt_legend_celltype)
fplot.plot_pca(pca_scores, 5, cell_color_vec= colors_dict_humanKidney['sex'],
               legend_handles=True,plt_legend_list=plt_legend_sex)
fplot.plot_pca(pca_scores, 3, cell_color_vec= colors_dict_humanKidney['sample'],
               legend_handles=True,plt_legend_list=plt_legend_sample)

#### plot the loadings of the factors
fplot.plot_factor_loading(pca_loading.T, genes, 0, 10, fontsize=10, 
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

num_pcs = 5
fplot.plot_pca(pca_scores_varimax, num_pcs, cell_color_vec= colors_dict_humanKidney['cell_type'],
               legend_handles=True,plt_legend_list=plt_legend_celltype)
fplot.plot_pca(pca_scores_varimax, num_pcs, cell_color_vec= colors_dict_humanKidney['sex'],
               legend_handles=True,plt_legend_list=plt_legend_sex)

fplot.plot_pca(pca_scores_varimax, num_pcs, cell_color_vec= colors_dict_humanKidney['sample'],
               legend_handles=True,plt_legend_list=plt_legend_sample)



fplot.plot_factor_loading(varimax_loading, genes, 0, 17, fontsize=10, 
                    num_gene_labels=2,title='Scatter plot of the loading vectors', 
                    label_x=True, label_y=True)

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
pca_scores_varimax_df_merged.to_csv('../Results/pca_scores_varimax_df_merged_kidneyMap.csv')
## save the varimax_loading_df and varimax_scores to a csv file
varimax_loading_df.to_csv('../Results/varimax_loading_df_kidneyMap.csv')




#### list the top 10 genes with the highest loading for each factor
for i in range(varimax_loading_df.shape[1]):
    print('factor: ', i)
    print('high loading genes: ')
    print(varimax_loading_df.iloc[:,i].sort_values(ascending=False)[0:10])
    print('-----------------------------')
    print('low loading genes: ')
    print(varimax_loading_df.iloc[:,i].sort_values(ascending=True)[0:10])

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
#rotation_results_promax = rot.promax_rotation(pca_loading.T)
#promax_loading = rotation_results_promax['rotloading']
#pca_scores_promax = rot.get_rotated_scores(pca_scores, rotation_results_promax['rotmat'])
#fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_humanKidney['strain'])
#fplot.plot_pca(pca_scores_promax, 9, cell_color_vec= colors_dict_humanKidney['cluster'])


########################
factor_loading = pca_loading
factor_scores = pca_scores
########################

factor_loading = rotation_results_varimax['rotloading']
factor_scores = pca_scores_varimax
########################
### find unique covariate levels
y_cell_type_unique = np.unique(y_cell_type)

####################################
#### Mean Importance score
####################################

### calculate the mean importance of each covariate level

mean_importance_df_sex = fmatch.get_mean_importance_all_levels(y_sex, factor_scores, time_eff=True,
                                                               scale='standard', mean='arithmatic')
mean_importance_df_cell_type = fmatch.get_mean_importance_all_levels(y_cell_type, factor_scores, time_eff=True,
                                                                     scale='standard', mean='arithmatic')

mean_importance_df_sample = fmatch.get_mean_importance_all_levels(y_sample, factor_scores, time_eff=True,
                                                                  scale='standard', mean='arithmatic')

### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
#mean_importance_df = pd.concat([mean_importance_df_sex, mean_importance_df_cell_type,mean_importance_df_sample], axis=0)
mean_importance_df = pd.concat([mean_importance_df_sex, mean_importance_df_cell_type], axis=0)

### remove the rows with NaN
mean_importance_df = mean_importance_df.dropna(axis=0)

mean_importance_df.shape
fplot.plot_all_factors_levels_df(mean_importance_df, title='F-C Match: Feature importance scores', color='coolwarm')
## getting rownnammes of the mean_importance_df
all_covariate_levels = mean_importance_df.index.values

##### Define global metrics for how well a factor analysis works on a dataset 
#### given a threshold for the feature importance scores, calculate the percentage of the factors that are matched with any covariate level
### plot the histogram of all the values in mean importance scores
### choosing a threshold for the feature importance scores
threshold = fmatch.get_otsu_threshold(mean_importance_df.values.flatten())

fplot.plot_histogram(mean_importance_df.values.flatten(), xlabel='Feature importance score',
                        title='Feature importance distribution', threshold=threshold,
                        save=False, save_path='./histogram_imp_kidneyMap.pdf',
                        xlabel_fontsize=21, ylabel_fontsize=20, title_fontsize=22,
                   xticks_fontsize=18,yticks_fontsize=16)

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
                                 title='', color='coolwarm', 
                                 x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                               x_axis_tick_fontsize=36, y_axis_tick_fontsize=38)


### save the plot to a pdf file without the function name
fplot.plot_all_factors_levels_df(mean_importance_df_matched, x_axis_label=x_labels_matched,
                                    title='', color='coolwarm',
                                    x_axis_fontsize=40, y_axis_fontsize=39, title_fontsize=40,
                                    x_axis_tick_fontsize=36, y_axis_tick_fontsize=38, 
                                    save=True, save_path='../Plots/mean_importance_df_matched_kidneyMap.pdf')

####################################
#### evaluating bimodality score using simulated factors ####
####################################

silhouette_scores = fmet.get_kmeans_scores(factor_scores, time_eff=True)
# silhouette_scores_gmm = fmet.get_gmm_scores(factor_scores, time_eff=True)
bimodality_index_scores = fmet.get_bimodality_index_all(factor_scores)
### calculate the average between the silhouette_scores_km, vrs_km and bimodality_index_scores
bimodality_scores = np.mean([silhouette_scores, bimodality_index_scores], axis=0)
bimodality_scores = bimodality_index_scores

#### Scaled variance
### calculate the number of levels in the covariate vectors cell_type sample and sex
num_levels_cell_type = len(np.unique(y_cell_type))
num_levels_sample = len(np.unique(y_sample))
num_levels_sex = len(np.unique(y_sex))
print('num_levels_cell_type: ', num_levels_cell_type)
print('num_levels_sample: ', num_levels_sample)
print('num_levels_sex: ', num_levels_sex)
### if num levels is more than 2, then use simpson index, if num levels is 2 then use arithmatic ASV 

#SV_all_factors = fmet.get_factors_SV_all_levels(factor_scores, y_cell_type) 
### label dependent factor metrics
ASV_all_arith_cell_type = fmet.get_ASV_all(factor_scores, covariate_vector=y_cell_type, mean_type='arithmetic')
ASV_all_arith_sex = fmet.get_ASV_all(factor_scores, y_sex, mean_type='arithmetic')
ASV_all_arith_sample = fmet.get_ASV_all(factor_scores, y_sample, mean_type='arithmetic')

### calculate diversity metrics
## simpson index: High scores (close to 1) indicate high diversity - meaning that the factor is not specific to any covariate level
## low simpson index (close to 0) indicate low diversity - meaning that the factor is specific to a covariate level
#factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df) ### calculated for the total importance matrix
#factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df_cell_type) ## calculated for each factor in the importance matrix
#factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  ## calculated for each factor in the importance matrix

factor_simpson_meanimp = fmet.get_all_factors_simpson_D_index(mean_importance_df) ## calculated for each factor in the importance matrix
factor_simpson_meanimp_cell_type = fmet.get_all_factors_simpson_D_index(mean_importance_df_cell_type) 
factor_simpson_meanimp_sample = fmet.get_all_factors_simpson_D_index(mean_importance_df_sample) 
factor_simpson_meanimp_sex = fmet.get_all_factors_simpson_D_index(mean_importance_df_sex) 



#### label free factor metrics
factor_variance_all = fmet.get_factor_variance_all(factor_scores)

### calculate 1-AUC_all_factors_df to measure the homogeneity of the factors
AUC_all_factors_df = fmet.get_AUC_all_factors_df(factor_scores, covariate_vector)
AUC_all_factors_df_1 = fmet.get_reversed_AUC_df(AUC_all_factors_df)
fplot.plot_all_factors_levels_df(AUC_all_factors_df_1,
                                    title='Homogeneity: 1-AUC scores', color='hot')




####################################
##### Factor metrics #####
####################################

all_metrics_dict = {'Bimodality':bimodality_scores, #bimodality_scores 
                    ## calculate 1 - factor_simpson_meanimp
                    'Specificity':factor_simpson_meanimp,
                    'Effect size': factor_variance_all,
                    'Homogeneity (cell type)':ASV_all_arith_cell_type,
                    'homogeneity (sex)':ASV_all_arith_sex,
                    'homogeneity (sample)':ASV_all_arith_sample}

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
                          title='Scaled metrics for all the factors', xticks_fontsize=30,
                           yticks_fontsize=33, legend_fontsize=25, save=True, 
                           save_path='../Plots/all_metrics_scaled_matched_kidneyMap_v2.pdf')

## subset x axis labels based on het matched factors
x_labels_matched = mean_importance_df_matched.columns.values



fplot.plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics)
fplot.plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics)

### convert all_metrics_scaled to a dataframe
all_metrics_scaled_df = pd.DataFrame(all_metrics_scaled)
all_metrics_scaled_df.columns = all_metrics_df.columns

### subset the all_metrics_df to the metrics_to_keep
metrics_to_keep = ['vrs_km', 'silhouette_gmm', 'bimodality_index', 
 'factor_variance', 'factor_specificity_meanimp', 'ASV_sample_geo', 'ASV_cell_type_geo']

all_metrics_df_sub = all_metrics_df[metrics_to_keep]
factor_metrics_sub = list(all_metrics_df_sub.columns)
all_metrics_scaled_sub = fmet.get_scaled_metrics(all_metrics_df_sub)


fplot.plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled metrics for all the factors')
fplot.plot_metric_heatmap(all_metrics_scaled_sub, factor_metrics_sub, title='Scaled metrics for all the factors')




fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=50)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=100)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=200)
fplot.plot_factor_dendogram(factor_scores, distance='ward',num_var=500)






'''
### randomly subsample the cells to 2000 cells
sample_size = 4000
subsample_index = np.random.choice(y.shape[0], size=sample_size, replace=False)
y = y[subsample_index,:]
data = data[subsample_index,:]
y_sample = y_sample[subsample_index]
y_sex = y_sex[subsample_index]
y_cell_type = y_cell_type[subsample_index]


### get the indices of the highly variable genes

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_RNA')


#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sampleID', data) 
x = np.column_stack((data.obs.nCount_RNA, x_sample)) 
x = sm.add_constant(x) ## adding the intercept

#### design matrix - sample only
x_sample = fproc.get_design_mat('sampleID', data) 
x = sm.add_constant(x_sample) ## adding the intercept


### fit GLM to each gene
glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('y shape: ', y.shape) # (num_cells, num_genes)
y = resid_pearson.T # (num_cells, num_genes)
print('y shape: ', y.shape) # (num_cells, num_genes)
'''
