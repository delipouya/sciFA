import numpy as np
import pandas as pd

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.graphics.api import abline_plot

import functions_metrics as fmet
import functions_plotting as fplot
import functions_processing as fproc
import functions_GLM as fglm
import functions_fc_match_classifier as fmatch

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


scores_mat_dict= {}
####################################
#### Running PCA on the data ######
####################################
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
scores_mat_dict['pca_scores']  = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_

####################################
#### Running PCA on the library size regressed pearon residual ######
####################################
#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_originalexp')
glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
scores_mat_dict['pca_lib_pearson']  = pipeline.fit_transform(resid_pearson.T)
pca = pipeline.named_steps['pca']
pca_loading_lib_pearson = pca.components_

####################################
#### Running PCA on the library size and protocol regressed pearon residual ######
####################################
#### design matrix - library size and sample
x_protocol = fproc.get_design_mat('protocol', data) 
x = np.column_stack((data.obs.nCount_originalexp, x_protocol)) 
x = sm.add_constant(x) ## adding the intercept

glm_fit_dict = fglm.fit_poisson_GLM(y, x)
resid_pearson = glm_fit_dict['resid_pearson'] 
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
scores_mat_dict['pca_lib_protocol_pearson']  = pipeline.fit_transform(resid_pearson.T)
pca = pipeline.named_steps['pca']
pca_loading_lib_protocol_pearson = pca.components_


################################################
#### Running varimax PCA on the library size regressed pearon residual  ######
################################################

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading_lib_pearson.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(scores_mat_dict['pca_lib_pearson'], rotation_results_varimax['rotmat'])
scores_mat_dict['varimax_lib_pearson']  = pca_scores_varimax


################################################
#### Running varimax PCA on the library size and protocol regressed pearon residual  ######
################################################

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading_lib_protocol_pearson.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(scores_mat_dict['pca_lib_protocol_pearson'], rotation_results_varimax['rotmat'])
scores_mat_dict['varimax_lib_protocol_pearson']  = pca_scores_varimax


################################################
############ calculating ASV for each model for comparison
################################################
ASV_protocol_all_arith_dict = {}
ASV_cell_line_all_arith_dict = {}
ASV_protocol_all_geo_dict = {}
ASV_cell_line_all_geo_dict = {}
factor_variance_all_dict = {}

vrs_km_dict = {}
silhouette_scores_gmm_dict = {}

for method in scores_mat_dict.keys():

    factor_scores = scores_mat_dict[method]
    ### label dependent factor metrics
    ASV_protocol_all_arith = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='arithmetic')
    ASV_cell_line_all_arith = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='arithmetic')
    ASV_protocol_all_geo = fmet.get_ASV_all(factor_scores, y_protocol, mean_type='geometric')
    ASV_cell_line_all_geo = fmet.get_ASV_all(factor_scores, y_cell_line, mean_type='geometric')


    ASV_protocol_all_arith_dict[method] = ASV_protocol_all_arith
    ASV_cell_line_all_arith_dict[method] = ASV_cell_line_all_arith
    ASV_protocol_all_geo_dict[method] = ASV_protocol_all_geo
    ASV_cell_line_all_geo_dict[method] = ASV_cell_line_all_geo

    ## variance
    factor_variance_all_dict[method] = fmet.get_factor_variance_all(factor_scores)

    ### bimoality metrics
    bic_scores_km, calinski_harabasz_scores_km, davies_bouldin_scores_km, silhouette_scores_km,\
        vrs_km, wvrs_km = fmet.get_kmeans_scores(factor_scores)
    bic_scores_gmm, silhouette_scores_gmm, vrs_gmm, wvrs_gmm = fmet.get_gmm_scores(factor_scores)

    vrs_km_dict[method] = vrs_km
    silhouette_scores_gmm_dict[method] = silhouette_scores_gmm


### converting to dataframe
fplot.plot_violinplot(pd.DataFrame.from_dict(ASV_protocol_all_arith_dict), fontsize=14, 
                ylab='ASV', title='arithmatic ASV for protocol')
fplot.plot_violinplot(pd.DataFrame.from_dict(ASV_cell_line_all_arith_dict), fontsize=14, 
                ylab='ASV', title='arithmatic ASV for cell line')

fplot.plot_violinplot(pd.DataFrame.from_dict(ASV_protocol_all_geo_dict), fontsize=14, 
                ylab='ASV', title='geometric ASV for protocol')
fplot.plot_violinplot(pd.DataFrame.from_dict(ASV_cell_line_all_geo_dict), fontsize=14, 
                ylab='ASV', title='geometric ASV for cell line')

fplot.plot_violinplot(pd.DataFrame.from_dict(factor_variance_all_dict), fontsize=14, 
                ylab='variance', title='Factor variance')

fplot.plot_violinplot(pd.DataFrame.from_dict(vrs_km_dict), fontsize=14,ylab='VRS (KM)', title='VRS for kmeans')      
fplot.plot_violinplot(pd.DataFrame.from_dict(silhouette_scores_gmm_dict), 
                      fontsize=14,ylab='silhouette score (GMM)', title='silhouette score for GMM')



################################################
mean_importance_df_dict = {}
threshold_dict = {}
factor_specificity_meanimp_dict = {}
for method in scores_mat_dict.keys():

    factor_scores = scores_mat_dict[method]
    ### calculate the mean importance of each covariate level
    mean_importance_df_protocol = fmatch.get_mean_importance_all_levels(y_protocol, factor_scores)
    mean_importance_df_cell_line = fmatch.get_mean_importance_all_levels(y_cell_line, factor_scores)
    ### concatenate mean_importance_df_protocol and mean_importance_df_cell_line
    mean_importance_df = pd.concat([mean_importance_df_protocol, mean_importance_df_cell_line], axis=0)
    mean_importance_df_dict[method] = mean_importance_df

    all_covariate_levels = mean_importance_df.index.values
    ### choosing a threshold for the feature importance scores
    threshold = fmatch.get_otsu_threshold(mean_importance_df.values.flatten())
    threshold_dict[method] = threshold

    ### calculate specificity
    factor_specificity_meanimp = fmet.get_all_factors_specificity(mean_importance_df)
    factor_specificity_meanimp_dict[method] = factor_specificity_meanimp


fplot.plot_violinplot(pd.DataFrame.from_dict(factor_specificity_meanimp_dict), 
                      fontsize=14,ylab='specificity', title='specificity (mean importance)')

### convert dictionaty of dataframe to dictionary of numpy arrays
mean_importance_mat_dict = {}
for method in mean_importance_df_dict.keys():
    mean_importance_df = mean_importance_df_dict[method]
    ###convert to flat numpy array
    mean_imp_flat = mean_importance_df.values.flatten()
    ### removing the values below the threshold 
    mean_imp_flat[mean_imp_flat<threshold_dict[method]] = 'NaN'
    #mean_imp_flat = mean_imp_flat[mean_imp_flat!=-1]
    mean_importance_mat_dict[method] = mean_imp_flat


fplot.plot_violinplot(pd.DataFrame.from_dict(mean_importance_mat_dict), 
                      fontsize=14,ylab='mean importance', title='mean importance (thresholded)')



#### repeat the procedure for AUC scores


####################################
#### AUC score
####################################
#### calculate the AUC of all the factors for all the covariate levels

AUC_df_dict = {}
threshold_dict_auc = {}
for method in scores_mat_dict.keys():

    factor_scores = scores_mat_dict[method]

    AUC_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
    AUC_all_factors_df_cell_line, wilcoxon_pvalue_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)

    AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
    AUC_df_dict[method] = AUC_all_factors_df

    threshold = fmatch.get_otsu_threshold(AUC_all_factors_df.values.flatten())
    threshold_dict_auc[method] = threshold




### convert dictionaty of dataframe to dictionary of numpy arrays
auc_mat_dict = {}
for method in AUC_df_dict.keys():
    auc_df = AUC_df_dict[method]
    ###convert to flat numpy array
    auc_flat = auc_df.values.flatten()
    ### removing the values below the threshold 
    auc_flat[auc_flat<threshold_dict_auc[method]] = 'NaN'
    auc_mat_dict[method] = auc_flat


fplot.plot_violinplot(pd.DataFrame.from_dict(auc_mat_dict), 
                      fontsize=14,ylab='AUC', title='AUC scores')


fplot.plot_violinplot(pd.DataFrame.from_dict(auc_mat_dict), 
                      fontsize=14,ylab='AUC', title='AUC scores(Thresholded)')
