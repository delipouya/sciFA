#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#### this script is used to evaluate the effect of different residual types on the importance scores
#### pearson, response and deviance residuals are used to calculate the importance scores
#### the importance scores are calculated for the scMix data in the baseline and the shuffled setting
# library size will be regressed out from the data

import sys
sys.path.append('./Code/')

import numpy as np
import pandas as pd

import functions_processing as fproc
import constants as const

import numpy as np
import pandas as pd
import time

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.graphics.api import abline_plot

import functions_plotting as fplot
import functions_processing as fproc
import functions_label_shuffle as flabel
import functions_metrics as fmet
import functions_fc_match_classifier as fmatch 
import functions_GLM as fglm

import rotations as rot
import constants as const
import statsmodels.api as sm


#### Function to concatenate mean importance dataframes specific to scMix data
def concatenate_meanimp_df(meanimp_df_list, mean_type_list, scale_type_list, scores_included_list):
    for i in range(len(meanimp_df_list)):
        meanimp_df_list[i]['mean_type'] = [mean_type_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['scale_type'] = [scale_type_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['scores_included'] = [scores_included_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['covariate'] = covariate_list
    meanimp_df = pd.concat(meanimp_df_list, axis=0)
    return meanimp_df




data = fproc.import_AnnData('/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad')
y_cell_line, y_sample, y_protocol = fproc.get_metadata_scMix(data)

data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()
y, num_cells, num_genes = fproc.get_data_array(data)
y = fproc.get_sub_data(y)
colors_dict_scMix = fplot.get_colors_dict_scMix(y_protocol, y_cell_line)
genes = data.var_names

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

#### extracting the pearson residuals, response residuals and deviance residuals
resid_pearson = glm_fit_dict['resid_pearson'] 
resid_response = glm_fit_dict['resid_response']
resid_deviance = glm_fit_dict['resid_deviance']

print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
print('response residuals: ', resid_response.shape) 
print('deviance residuals: ', resid_deviance.shape) 

### make a dictionary of residuals
resid_dict = {'pearson': resid_pearson, 'response': resid_response, 'deviance': resid_deviance}
### make a for loop to calculate the importance scores for each residual type
importance_df_dict = {}
time_dict_a_level_dict = {}
auc_df_dict = {}

for resid_type in resid_dict.keys():
    print('--'*20)
    print('residual type: ', resid_type)
    resid = resid_dict[resid_type]

    ### using pipeline to scale the gene expression data first

    print('y shape: ', y.shape) # (num_cells, num_genes)
    y = resid.T # (num_cells, num_genes)
    print('y shape: ', y.shape) # (num_cells, num_genes)

    ### using pipeline to scale the gene expression data first
    pipeline = Pipeline([('scaling', StandardScaler()), 
                         ('pca', PCA(n_components=const.num_components))])
    pca_scores = pipeline.fit_transform(y)
    pca = pipeline.named_steps['pca']
    pca_loading = pca.components_

    ####### Applying varimax rotation to the factor scores
    rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
    varimax_loading = rotation_results_varimax['rotloading']
    pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

    factor_scores = pca_scores_varimax

    FA_type = 'varimax' #'PCA'
    scores_included = 'baseline' #'baseline'#'top_cov' 'top_FA' 
    n = 100

    ####################################
    #### Baseline importance calculation and run time ######
    #### Importance calculation and run time as baseline ######

    importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol, factor_scores) 
    importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line, factor_scores) 

    meanimp_standard_arith_protocol, meanimp_standard_geom_protocol, meanimp_minmax_arith_protocol, meanimp_minmax_geom_protocol = flabel.get_mean_importance_df_v_comp(importance_df_dict_protocol)
    meanimp_standard_arith_cell_line, meanimp_standard_geom_cell_line, meanimp_minmax_arith_cell_line, meanimp_minmax_geom_cell_line = flabel.get_mean_importance_df_v_comp(importance_df_dict_cell_line)

    meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_protocol, meanimp_standard_arith_cell_line], axis=0)
    meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_protocol, meanimp_standard_geom_cell_line], axis=0)
    meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_protocol, meanimp_minmax_arith_cell_line], axis=0)
    meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_protocol, meanimp_minmax_geom_cell_line], axis=0)


    ####################################
    #### AUC score
    ####################################
    #### calculate the AUC of all the factors for all the covariate levels
    AUC_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
    AUC_all_factors_df_cell_line, wilcoxon_pvalue_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)

    AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
    #auc_df_dict[resid_type] = AUC_all_factors_df

    meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                    meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                    AUC_all_factors_df]

    mean_type_list = ['arithmatic', 'geometric', 'arithmatic', 'geometric', 'AUC']

    scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'AUC']

    scores_included_list = [scores_included]*5
    covariate_list = ['protocol']*3 + ['cell_line']*3

    meanimp_df = concatenate_meanimp_df(meanimp_df_list, mean_type_list, scale_type_list, scores_included_list)
    meanimp_df.to_csv('/home/delaram/sciFA/Results/residual/meanimp_df_'+'scMixology_'+ 
                      FA_type+'_'+scores_included+'_'+resid_type + '.csv')
    


    importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
    importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
    ### save importance_df_m and meanimp_df to csv
    importance_df_m.to_csv('/home/delaram/sciFA/Results/residual/importance_df_melted_'+'scMixology_'+ 
                      FA_type+'_'+scores_included+'_'+resid_type + '.csv')

    
    #### shuffle the covariate vectors n times in a loop
    scores_included = 'shuffle' 
    t_start_total = time.time()
    for i in range(n):
        print('i: ', i)

        y_protocol_shuffled = flabel.shuffle_covariate(y_protocol)
        y_cell_line_shuffled = flabel.shuffle_covariate(y_cell_line)

        ####################################
        #### Importance calculation and run time for model comparison  ######
        ####################################
        importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol_shuffled, factor_scores) 
        importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line_shuffled, factor_scores) 
        
    
        meanimp_standard_arith_protocol, meanimp_standard_geom_protocol, meanimp_minmax_arith_protocol, meanimp_minmax_geom_protocol = flabel.get_mean_importance_df_v_comp(importance_df_dict_protocol)
        meanimp_standard_arith_cell_line, meanimp_standard_geom_cell_line, meanimp_minmax_arith_cell_line, meanimp_minmax_geom_cell_line = flabel.get_mean_importance_df_v_comp(importance_df_dict_cell_line)

        meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_protocol, meanimp_standard_arith_cell_line], axis=0)
        meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_protocol, meanimp_standard_geom_cell_line], axis=0)
        meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_protocol, meanimp_minmax_arith_cell_line], axis=0)
        meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_protocol, meanimp_minmax_geom_cell_line], axis=0)

        ####################################
        #### AUC score
        ####################################
        #### calculate the AUC of all the factors for all the covariate levels
        AUC_all_factors_df_protocol, wilcoxon_pvalue_all_factors_df_protocol = fmet.get_AUC_all_factors_df(factor_scores, y_protocol)
        AUC_all_factors_df_cell_line, wilcoxon_pvalue_all_factors_df_cell_line = fmet.get_AUC_all_factors_df(factor_scores, y_cell_line)

        AUC_all_factors_df = pd.concat([AUC_all_factors_df_protocol, AUC_all_factors_df_cell_line], axis=0)
        #auc_df_dict[resid_type] = AUC_all_factors_df

        meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                        meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                        AUC_all_factors_df]

        mean_type_list = ['arithmatic', 'geometric', 'arithmatic', 'geometric', 'AUC']

        scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'AUC']

        scores_included_list = [scores_included]*5
        covariate_list = ['protocol']*3 + ['cell_line']*3

        meanimp_df = concatenate_meanimp_df(meanimp_df_list, mean_type_list, scale_type_list, scores_included_list)
        meanimp_df.to_csv('/home/delaram/sciFA/Results/residual/'+resid_type+'/meanimp_df_'+'scMixology_'+ 
                        FA_type+'_'+scores_included+'_'+resid_type+'_'+str(i) + '.csv')
        

        ############################################################
        ########## Comparing factor scores between models
        ############################################################
        #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
        importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
        importance_df_m = flabel.get_melted_importance_df(importance_df_dict)

        ### save importance_df_m to csv
        importance_df_m.to_csv('/home/delaram/sciFA/Results/residual/'+resid_type+
                               '/importance_df_melted_'+'scMixology_'+ 
                               FA_type+'_'+scores_included+'_'+resid_type+'_'+str(i) + '.csv')
    

    t_end_total = time.time()
    print('Total time: ', t_end_total - t_start_total)
