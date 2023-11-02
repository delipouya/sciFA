#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import functions_processing as fproc
import constants as const

import numpy as np
import pandas as pd

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.graphics.api import abline_plot

import functions_plotting as fplot
import functions_processing as fproc
import functions_label_shuffle as flabel

import rotations as rot
import constants as const

np.random.seed(10)
import time



######################################################################################
###################################################################################

data = fproc.import_AnnData('/home/delaram/sciFA/Data/scMix_3cl_merged.h5ad')
y_cell_line, y_sample, y_protocol = fproc.get_metadata_scMix(data)

data.obs['protocol'] = y_protocol.to_numpy()
data.obs['cell_line'] = y_cell_line.to_numpy()
data.obs['sample'] = y_sample.to_numpy()
y, num_cells, num_genes = fproc.get_data_array(data)
y = fproc.get_sub_data(y)
colors_dict_scMix = fplot.get_colors_dict_scMix(y_protocol, y_cell_line)
genes = data.var_names


####################################
#### Running PCA on the data ######
####################################
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=const.num_components))])
pca_scores = pipeline.fit_transform(y)
pca = pipeline.named_steps['pca']
pca_loading = pca.components_
pca_loading.shape #(factors, genes)

factor_scores = pca_scores

######## Applying varimax rotation to the factor scores
rotation_results_varimax = rot.varimax_rotation(pca_loading.T)
varimax_loading = rotation_results_varimax['rotloading']
pca_scores_varimax = rot.get_rotated_scores(pca_scores, rotation_results_varimax['rotmat'])

factor_scores = pca_scores_varimax



FA_type = 'varimax'#'PCA'
scores_included = 'baseline'#'baseline'#'top_cov' 'top_FA' 
n = 1000

####################################
#### Baseline importance calculation and run time ######
#### Importance calculation and run time as baseline ######
importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol, factor_scores) 
importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line, factor_scores) 
########### Comparing model run times
time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
########## Comparing time differences between models
time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
#plot_runtime_barplot(time_df)

########## Comparing factor scores between models
#### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
importance_df_m = flabel.get_melted_importance_df(importance_df_dict)


### save importance_df_m to csv
importance_df_m.to_csv('/home/delaram/sciFA/Results/importance_df_melted_'+
                        'scMixology_'+FA_type+'_'+'baseline_n1000.csv')


t_start_total = time.time()
#### shuffle the covariate vectors n times in a loop
for i in range(n):
    print('i: ', i)

    y_protocol_shuffled = flabel.shuffle_covariate(y_protocol)
    y_cell_line_shuffled = flabel.shuffle_covariate(y_cell_line)

    ####################################
    #### Importance calculation and run time for model comparison  ######
    ####################################
    importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol_shuffled, factor_scores) 
    importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line_shuffled, factor_scores) 
    
    ####################################
    ########### Comparing model run times
    ####################################
    time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
    #plot_runtime_barplot(time_df)
    ### save time_df to csv
    time_df.to_csv('/home/delaram/sciFA/Results/shuffle_empirical_dist_time/time_df_'+
                   'scMixology_'+FA_type+'_'+'shuffle_'+str(i)+'.csv')


    ############################################################
    ########## Comparing factor scores between models
    ############################################################
    #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
    importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
    importance_df_m = flabel.get_melted_importance_df(importance_df_dict)

    ### save importance_df_m to csv
    importance_df_m.to_csv('/home/delaram/sciFA/Results/shuffle_empirical_dist/importance_df_melted_'+
                           'scMixology_'+FA_type+'_'+'shuffle_'+str(i)+'.csv')

t_end_total = time.time()
print('Total time: ', t_end_total - t_start_total)