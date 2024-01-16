#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import functions_fc_match_classifier as fmatch 

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
import time

np.random.seed(10)

#### Function to concatenate mean importance dataframes specific to scMix data
def concatenate_meanimp_df(meanimp_df_list, mean_type_list, scale_type_list, 
                           scores_included_list, residual_type, covariate_list):
    for i in range(len(meanimp_df_list)):
        meanimp_df_list[i]['mean_type'] = [mean_type_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['scale_type'] = [scale_type_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['scores_included'] = [scores_included_list[i]]*meanimp_df_list[i].shape[0]
        meanimp_df_list[i]['covariate'] = covariate_list
    meanimp_df = pd.concat(meanimp_df_list, axis=0)
    ### add a column for resiudal type name
    meanimp_df['residual_type'] = [residual_type]*meanimp_df.shape[0]
    return meanimp_df



data_file_path = '/home/delaram/sciFA//Data/HumanLiverAtlas.h5ad'
data = fproc.import_AnnData(data_file_path)

data, gene_idx = fproc.get_sub_data(data, random=False) # subset the data to num_genes HVGs
y, genes, num_cells, num_genes = fproc.get_data_array(data)
y_sample, y_cell_type = fproc.get_metadata_humanLiver(data)
genes = data.var_names

### calculate row and column sums of y
rowsum_y = np.sum(y, axis=1)
colsum_y = np.sum(y, axis=0)


#### design matrix - library size only
#x = fproc.get_lib_designmat(data, lib_size='total_counts')

#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sample', data) 
x = np.column_stack((data.obs.total_counts, x_sample)) 
x = sm.add_constant(x) ## adding the intercept

num_levels_sample = len(y_sample.unique())
num_levels_cell_type = len(y_cell_type.unique())
####################################
#### fit GLM to each gene ######
####################################

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

    importance_df_dict_sample, time_dict_a_level_dict_sample = flabel.get_importance_all_levels_dict(y_sample, factor_scores, time_eff=True) 
    importance_df_dict_cell_type, time_dict_a_level_dict_cell_type = flabel.get_importance_all_levels_dict(y_cell_type, factor_scores, time_eff=True) 

    meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = flabel.get_mean_importance_df_list(importance_df_dict_sample)
    meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = flabel.get_mean_importance_df_list(importance_df_dict_cell_type)

    meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type], axis=0)
    meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type], axis=0)
    meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type], axis=0)
    meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type], axis=0)
    meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type], axis=0)
    meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type], axis=0)


    meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                    meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                    meanimp_rank_arith_df, meanimp_rank_geom_df]

    mean_type_list = ['arithmatic', 'geometric', 
                    'arithmatic', 'geometric', 
                    'arithmatic', 'geometric']

    scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

    scores_included_list = [scores_included]*len(meanimp_df_list)
    covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type

    meanimp_df = concatenate_meanimp_df(meanimp_df_list, mean_type_list, 
                                        scale_type_list, scores_included_list, 
                                        residual_type=residual_type, covariate_list=covariate_list)


    ############################################################
    ########### Comparing model run times
    time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', 
                                    columns=list(time_df_dict.values())[0].keys())
    #flabel.plot_runtime_barplot(time_df)

    ########## Comparing factor scores between models
    #### merge two importance_df_dict_sample and importance_df_dict_cell_type dicts
    importance_df_dict = {**importance_df_dict_sample, **importance_df_dict_cell_type}
    importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
    ### add a column for residual type name to importance_df_m
    importance_df_m['residual_type'] = [residual_type]*importance_df_m.shape[0]


    ### save importance_df_m and meanimp_df to csv
    importance_df_m.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/base/'+'importance_df_melted_human_liver_'+residual_type+'_'+'baseline.csv')
    meanimp_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/base/'+'meanimp_df_'+'human_liver_'+residual_type+'_'+'baseline.csv')

    t_start_total = time.time()
    #### shuffle the covariate vectors n times in a loop
    for i in range(n):
        print('i: ', i)

        y_sample_shuffled = flabel.shuffle_covariate(y_sample)
        y_cell_type_shuffled = flabel.shuffle_covariate(y_cell_type)

        ####################################
        #### Importance calculation and run time for model comparison  ######
        ####################################
        importance_df_dict_sample, time_dict_a_level_dict_sample = flabel.get_importance_all_levels_dict(y_sample_shuffled, factor_scores, time_eff=True) 
        importance_df_dict_cell_type, time_dict_a_level_dict_cell_type = flabel.get_importance_all_levels_dict(y_cell_type_shuffled, factor_scores, time_eff=True) 
        
        ####################################
        ########### Comparing model run times
        ####################################
        time_df_dict = {**time_dict_a_level_dict_sample, **time_dict_a_level_dict_cell_type}
        ########## Comparing time differences between models
        time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
        #plot_runtime_barplot(time_df)
        ### save time_df to csv
        time_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+
                       '/time/' + 'time_df_human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        
        ####################################
        ##### Mean importance calculation ########
        meanimp_standard_arith_sample, meanimp_standard_geom_sample, meanimp_minmax_arith_sample, meanimp_minmax_geom_sample, meanimp_rank_arith_sample, meanimp_rank_geom_sample = flabel.get_mean_importance_df_list(importance_df_dict_sample)
        meanimp_standard_arith_cell_type, meanimp_standard_geom_cell_type, meanimp_minmax_arith_cell_type, meanimp_minmax_geom_cell_type, meanimp_rank_arith_cell_type, meanimp_rank_geom_cell_type = flabel.get_mean_importance_df_list(importance_df_dict_cell_type)


        meanimp_standard_arith_df = pd.concat([meanimp_standard_arith_sample, meanimp_standard_arith_cell_type], axis=0)
        meanimp_standard_geom_df = pd.concat([meanimp_standard_geom_sample, meanimp_standard_geom_cell_type], axis=0)
        meanimp_minmax_arith_df = pd.concat([meanimp_minmax_arith_sample, meanimp_minmax_arith_cell_type], axis=0)
        meanimp_minmax_geom_df = pd.concat([meanimp_minmax_geom_sample, meanimp_minmax_geom_cell_type], axis=0)
        meanimp_rank_arith_df = pd.concat([meanimp_rank_arith_sample, meanimp_rank_arith_cell_type], axis=0)
        meanimp_rank_geom_df = pd.concat([meanimp_rank_geom_sample, meanimp_rank_geom_cell_type], axis=0)


        meanimp_df_list = [meanimp_standard_arith_df, meanimp_standard_geom_df, 
                        meanimp_minmax_arith_df, meanimp_minmax_geom_df, 
                        meanimp_rank_arith_df, meanimp_rank_geom_df]

        mean_type_list = ['arithmatic', 'geometric', 
                        'arithmatic', 'geometric', 
                        'arithmatic', 'geometric']

        scale_type_list = ['standard', 'standard', 'minmax', 'minmax', 'rank', 'rank']

        scores_included = 'shuffle'#'baseline'#'top_cov' 'top_FA' 
        scores_included_list = [scores_included]*len(meanimp_df_list)
        covariate_list = ['sample']*num_levels_sample + ['cell_type']*num_levels_cell_type

        meanimp_df = concatenate_meanimp_df(meanimp_df_list, mean_type_list, 
                                            scale_type_list, scores_included_list, 
                                            residual_type=residual_type, covariate_list=covariate_list)

        ############################################################
        ########## Comparing factor scores between models
        ############################################################
        #### merge two importance_df_dict_sample and importance_df_dict_cell_type dicts
        importance_df_dict = {**importance_df_dict_sample, **importance_df_dict_cell_type}
        importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
        ### add a column for residual type name to importance_df_m
        importance_df_m['residual_type'] = [residual_type]*importance_df_m.shape[0]


        ### save importance_df_m and meanimp_df to csv
        importance_df_m.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/imp/'+
                               'importance_df_melted_human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')
        meanimp_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/meanimp/'+
                          'meanimp_df_'+'human_liver_'+residual_type+'_'+'shuffle_'+str(i)+'.csv')


    t_end_total = time.time()
    print('Total time: ', t_end_total - t_start_total)


