#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import functions_processing as fproc
import constants as const

import numpy as np
import pandas as pd
import functions_GLM as fglm

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.graphics.api import abline_plot

import functions_plotting as fplot
import functions_processing as fproc
import functions_label_shuffle as flabel
import functions_metrics as fmet
import statsmodels.api as sm

import rotations as rot
import constants as const

np.random.seed(10)
import time


######################################################################################
###################################################################################

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
glm_fit_dict = fglm.fit_poisson_GLM(y, x)

#### extracting the pearson residuals, response residuals and deviance residuals
resid_pearson = glm_fit_dict['resid_pearson'] 
print('pearson residuals: ', resid_pearson.shape) # numpy array of shape (num_genes, num_cells)
resid_dict = {'pearson': resid_pearson}

### make a for loop to calculate the importance scores for each residual type
importance_df_dict = {}
time_dict_a_level_dict = {}


for residual_type in resid_dict.keys():
    print('--'*20)
    print('residual type: ', residual_type)
    resid = resid_dict[residual_type]

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
    scores_included = 'baseline'#'baseline'#'top_cov' 'top_FA' 
    #n = 1000
    n = 500

    ####################################
    #### Baseline importance calculation and run time ######
    #### Importance calculation and run time as baseline ######

    importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol, factor_scores, time_eff=False) 
    importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line, factor_scores, time_eff=False) 

    covariate_list = ['protocol']*3 + ['cell_line']*3


    ##########################################################
    ########### Comparing model run times
    time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                    columns=list(time_df_dict.values())[0].keys())
    #flabel.plot_runtime_barplot(time_df)

    ########## Comparing factor scores between models
    #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
    importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
    importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
    ### add a column for residual type name to importance_df_m
    importance_df_m['residual_type'] = [residual_type]*importance_df_m.shape[0]


    ### save importance_df_m and meanimp_df to csv
    importance_df_m.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+'/base/'+'importance_df_melted_scMixology_'+residual_type+'_'+'baseline.csv')

    t_start_total = time.time()
    #### shuffle the covariate vectors n times in a loop
    for i in range(n):
        print('i: ', i)

        y_protocol_shuffled = flabel.shuffle_covariate(y_protocol)
        y_cell_line_shuffled = flabel.shuffle_covariate(y_cell_line)

        ####################################
        #### Importance calculation and run time for model comparison  ######
        ####################################
        importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol_shuffled, factor_scores, time_eff=False) 
        importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line_shuffled, factor_scores, time_eff=False) 
        
        ####################################
        ########### Comparing model run times
        ####################################
        time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
        ########## Comparing time differences between models
        time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
        #plot_runtime_barplot(time_df)
        ### save time_df to csv
        time_df.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+
                       '/time/' + 'time_df_scMixology_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        
        ############################################################
        ########## Comparing factor scores between models
        ############################################################
        #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
        importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
        importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
        ### add a column for residual type name to importance_df_m
        importance_df_m['residual_type'] = [residual_type]*importance_df_m.shape[0]


        ### save importance_df_m and meanimp_df to csv
        importance_df_m.to_csv('/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/imp/'+
                               'importance_df_melted_scMixology_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')

    t_end_total = time.time()
    print('Total time: ', t_end_total - t_start_total)