import sys 
sys.path.append('./Code/')
import numpy as np
import pandas as pd

import functions_processing as fproc
import constants as const

import numpy as np
import pandas as pd
from scipy.io import mmread

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
import statsmodels.api as sm

np.random.seed(10)


######################################################################################
###################################################################################

data = fproc.import_AnnData('./Data/scMix_3cl_merged.h5ad')
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
n = 10

####################################
#### Baseline importance calculation and run time ######
#### Importance calculation and run time as baseline ######
importance_df_dict_protocol, time_dict_a_level_dict_protocol = flabel.get_importance_all_levels_dict(y_protocol, factor_scores) 
importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = flabel.get_importance_all_levels_dict(y_cell_line, factor_scores) 
########### Comparing model run times
time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
########## Comparing time differences between models
time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
flabel.plot_runtime_barplot(time_df)

########## Comparing factor scores between models
#### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
importance_df_m = flabel.get_melted_importance_df(importance_df_dict)
flabel.plot_importance_boxplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison ' + FA_type))
flabel.plot_importance_violinplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
flabel.plot_importance_boxplot(importance_df_m, x='model', y='importance',title='Model score comparison '+ FA_type)
flabel.plot_importance_violinplot(importance_df_m, x='model', y='importance',title='Model score comparison '+ FA_type)


### save importance_df_m to csv
importance_df_m.to_csv('./Results/importance_df_melted_'+
                        'scMixology_'+FA_type+'_'+'baseline.csv')



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
    flabel.plot_runtime_barplot(time_df)
    

    ############################################################
    ########## Comparing factor scores between models
    ############################################################
    #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
    importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
    importance_df_m = flabel.get_melted_importance_df(importance_df_dict)

    
    flabel.plot_importance_boxplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
    flabel.plot_importance_violinplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
    
    flabel.plot_importance_boxplot(importance_df_m, x='model', y='importance',title='Model score comparison'+ FA_type)
    flabel.plot_importance_violinplot(importance_df_m, x='model', y='importance',title='Model score comparison'+ FA_type)

    ### save importance_df_m to csv
    importance_df_m.to_csv('./Results/importance_df_melted_'+
                           'scMixology_'+FA_type+'_'+'shuffle_v'+str(i)+'.csv')




#######################################################
##### read all the csv files in './Results/importance_df_melted_scMixology_varimax_shuffle_results/' and concatenate them into one dataframe
#######################################################
importance_df_m_merged = pd.DataFrame()
for i in range(n):
    print('i: ', i)
    importance_df_m = pd.read_csv('./Results/importance_df_melted_scMixology_varimax_shuffle_results/importance_df_melted_scMixology_varimax_shuffle_v'+str(i)+'.csv')
    importance_df_m['shuffle'] = np.repeat('shuffle_'+str(i), importance_df_m.shape[0])
    importance_df_m_merged = pd.concat([importance_df_m_merged, importance_df_m], axis=0)
    

#### read the importance_df_melted_scMixology_varimax_baseline.csv file and concatenate it to importance_df_m_merged
importance_df_m_baseline = pd.read_csv('./Results/importance_df_melted_scMixology_varimax_baseline.csv')
importance_df_m_baseline['shuffle'] = np.repeat('baseline', importance_df_m_baseline.shape[0])
importance_df_m_merged = pd.concat([importance_df_m_merged, importance_df_m_baseline], axis=0)
### drop the Unnamed: 0 column
importance_df_m_merged.drop(columns=['Unnamed: 0'], inplace=True)

### reorder shuffle column as baseline, shuffle_0, shuffle_1, ... for visualization
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].astype('category')
importance_df_m_merged['shuffle'].cat.reorder_categories(['baseline'] + ['shuffle_'+str(i) for i in range(n)], inplace=True)
importance_df_m_merged.head()


flabel.plot_importance_boxplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
flabel.plot_importance_violinplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')


############# replace shuffle_0, shuffle_1, ... with shuffle
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].replace(['shuffle_'+str(i) for i in range(n)], 
                                                                              ['shuffle' for i in range(n)])
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].astype('category')
importance_df_m_merged['shuffle'].cat.reorder_categories(['baseline', 'shuffle'], inplace=True)


flabel.plot_importance_boxplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
flabel.plot_importance_violinplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')



############# selecting the top importance score for each model and each factor
### split the importance_df_m into dataframes based on "model" and for each 'factor' only keep the top score
importance_df_m_merged_top = importance_df_m_merged.groupby(['model', 'factor', 'shuffle']).apply(lambda x: x.nlargest(1, 'importance')).reset_index(drop=True)
### calculate the mean and sd importance of each model for baseline annd shuffle categories
importance_df_m_merged_mean = importance_df_m_merged_top.groupby(['model', 'shuffle']).mean().reset_index()
importance_df_m_merged_std = importance_df_m_merged_top.groupby(['model', 'shuffle']).std().reset_index()
### split importance_df_m_merged_mean based on model to a dictionary,with model as key and suffle and baseline as values
importance_df_m_merged_mean_dict = dict(tuple(importance_df_m_merged_mean.groupby('model')))
importance_df_m_merged_std_dict = dict(tuple(importance_df_m_merged_std.groupby('model')))

imp_drop_score_dict = {}
imp_mean_drop_dict = {}

## select the first model in importance_df_m_merged_mean_dict.keys()

for a_model in list(importance_df_m_merged_mean_dict.keys()):
    mean_l = list(importance_df_m_merged_mean_dict[a_model]['importance'])
    sd_l = list(importance_df_m_merged_std_dict[a_model]['importance'])
    numerator = mean_l[0] - mean_l[1]
    #denominator = np.sqrt(sd_l[0] * sd_l[1]) 
    denominator = sd_l[0] * sd_l[1]
    imp_mean_drop_dict[a_model] = numerator
    imp_drop_score_dict[a_model] = numerator/denominator

print(imp_mean_drop_dict)
print(imp_drop_score_dict)


### make a gourped violin plot of importance_df_m_top using sns and put the legend outside the plot
### boxplot of importance_df_m_merged using sns, shuffle is the hue, model as x axis
### put baseline as the first boxplot



############# replace shuffle_0, shuffle_1, ... with shuffle
importance_df_m_merged_top['shuffle'] = importance_df_m_merged_top['shuffle'].replace(['shuffle_'+str(i) for i in range(n)], 
                                                                              ['shuffle' for i in range(n)])
importance_df_m_merged_top['shuffle'] = importance_df_m_merged_top['shuffle'].astype('category')
importance_df_m_merged_top['shuffle'].cat.reorder_categories(['baseline', 'shuffle'], inplace=True)

flabel.plot_importance_boxplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
flabel.plot_importance_violinplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')

