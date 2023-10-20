
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns

import functions_processing as fproc
import constants as const

import skimage as ski
import skimage.filters as skif


import random
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
import time

# training classifiers for feature importance on a classification problem
# matching pca factors to different covariates in the data

def get_importance_df_v_comp(factor_scores, a_binary_cov) -> pd.DataFrame:
    '''
    calculate the importance of each factor for each covariate level
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    a_binary_cov: numpy array of the binary covariate for a covariate level (n_cells, )
    '''

    models = {'LogisticRegression': LogisticRegression(), 
              'DecisionTree': DecisionTreeClassifier(), 
              'RandomForest': RandomForestClassifier(), 
              'XGB': XGBClassifier(), 
              'KNeighbors_permute': KNeighborsClassifier()}

    importance_dict = {}
    ### save the time of the fit for each model
    time_dict = {}

    for model_name, model in models.items():
        X, y = factor_scores, a_binary_cov

        t_start= time.time()
        model.fit(X, y)
        t_end = time.time()
        time_dict[model_name] = t_end - t_start

        if model_name == 'LogisticRegression':
            importance_dict[model_name] = model.coef_[0]

        elif model_name in ['DecisionTree', 'RandomForest', 'XGB']:
            # get importance
            importance_dict[model_name] = model.feature_importances_
        else:
            # perform permutation importance
            results = permutation_importance(model, X, y, scoring='accuracy')
            importance_dict[model_name] = results.importances_mean

    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', 
                                           columns=['F'+str(i) for i in range(1, factor_scores.shape[1]+1)])
    return importance_df, time_dict



def get_importance_all_levels_dict(covariate_vec, factor_scores) -> pd.DataFrame:
    '''
    calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, num_components)
    covariate_vec: numpy array of the covariate vector (n_cells, )
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''

    importance_df_a_level_dict = {}
    time_dict_a_level_dict = {}

    for covariate_level in np.unique(covariate_vec):
        print('covariate_level: ', covariate_level)

        a_binary_cov = fproc.get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level, time_dict = get_importance_df_v_comp(factor_scores, a_binary_cov)
        
        importance_df_a_level_dict[covariate_level] = importance_df_a_level
        time_dict_a_level_dict[covariate_level] = time_dict


    return importance_df_a_level_dict, time_dict_a_level_dict


###################################################################################
###################### Mean calculation for importance ############################
###################################################################################
def get_mean_importance_level(importance_df_a_level) -> np.array:
    ''' 
    calculate the mean importance of one level of a given covariate and returns a vector of length of number of factors
    importance_df_a_level: a dataframe of the importance of each factor for a given covariate level
    '''
    importance_df_np = np.asarray(importance_df_a_level)
    ### scale each row of the importance_df_np to be positive
    importance_df_np = importance_df_np - importance_df_np.min(axis=1, keepdims=True)
    ### normalize each row of the importance_df_np to be between 0 and 1
    importance_df_np = importance_df_np / importance_df_np.max(axis=1, keepdims=True)
    ### calculate the mean of each column of the importance_df_np
    mean_importance = np.mean(importance_df_np, axis=0)
    return mean_importance


def get_mean_importance_all_levels(covariate_vec, factor_scores) -> pd.DataFrame:
    '''
    calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, num_components)
    covariate_vec: numpy array of the covariate vector (n_cells, )
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''

    importance_df_a_level_dict = {}

    mean_importance_df = pd.DataFrame(columns=['PC'+str(i) for i in range(1, factor_scores.shape[1]+1)])

    for covariate_level in np.unique(covariate_vec):
        print('covariate_level: ', covariate_level)

        a_binary_cov = fproc.get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level = fmatch.get_importance_df(factor_scores, a_binary_cov)
        mean_importance_a_level = get_mean_importance_level(importance_df_a_level)
        importance_df_a_level_dict[covariate_level] = importance_df_a_level

        print('mean_importance_a_level:', mean_importance_a_level)
        mean_importance_df.loc[covariate_level] = mean_importance_a_level

    return mean_importance_df


def shuffle_covariate(covariate_vector) -> np.array:
    '''
    shuffle the covariate vector and return the shuffled covariate vector
    covariate_vector: numpy array of the covariate vector (n_cells, )
    '''
    covariate_vector_shuffled = np.copy(covariate_vector)
    np.random.shuffle(covariate_vector_shuffled)
    return covariate_vector_shuffled
    
    



def plot_runtime_barplot(time_df):
    ''' make a barplot of time_df using sns and put the legend outside the plot
    x tick labels are models. each bar is a covariate level
    time_df: a dataframe of size (num_levels, num_models)
    '''
    plt.figure(figsize=(10, 7))
    ### get row names of time_df_protocol as numpy array
    row_names = time_df.index.to_numpy()
    time_df_m = time_df.melt(var_name='model', value_name='time')
    time_df_m['covariate_level'] = np.resize(row_names,time_df_m.shape[0])
    #set seaborn plotting aesthetics
    sns.set(style='white')
    #create grouped bar chart with x-tick labels rotated 90 degrees
    sns.barplot(x='model', y='time', hue='covariate_level', data=time_df_m)
    plt.title('Model run time comparison' + ' (' + FA_type + ')')
    plt.xticks(rotation=45)



def get_melted_importance_df(importance_df_dict) -> pd.DataFrame:
    '''
    melt the importance_df_dict and return a dataframe of size (num_levels*num_models, num_components)
    importance_df_dict: a dictionary of dataframes of the importance of each factor for a given covariate level
    '''
    ### make a boxplot of importance_df_dict
    importance_df = pd.concat(importance_df_dict.values(), keys=importance_df_dict.keys())
    importance_df.index.names = ['covariate_level', 'model']

    ### get row names of importance_df column covariate_level
    row_names_cov = importance_df.index.get_level_values('covariate_level').to_numpy()
    row_names_model = importance_df.index.get_level_values('model').to_numpy()

    ### melt the importance_df
    importance_df_m = importance_df.melt(var_name='factor', value_name='importance')
    ### add covariate_level column to importance_df_m
    importance_df_m['model'] = np.resize(row_names_model, importance_df_m.shape[0])
    importance_df_m['covariate_level'] = np.resize(row_names_cov, importance_df_m.shape[0])

    return importance_df_m



def plot_importance_boxplot(importance_df_m, 
                            x, y, hue=None,xtick_fontsize=15, ytick_fontsize=15, 
                            title='Model score comparison', xlab=''):
    plt.figure(figsize=(6, 3))
    sns.set(style='white')
    if hue is None:
        sns.boxplot(x=x, y=y, data=importance_df_m)
    else:
        sns.boxplot(x=x, y=y, hue=hue, data=importance_df_m)
    plt.xticks(rotation=45, fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xlabel(xlab,fontsize=20)
    plt.ylabel('Importance score', fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)


def plot_importance_violinplot(importance_df_m, 
                               x, y, hue=None, xtick_fontsize=15, ytick_fontsize=15,
                               title='Model score comparison', xlab=''):
    ''' make a merged violinplot of importance_df_m using sns and put the legend outside the plot
    x tick and each violinplot is a model's importance score for all factor
    importance_df_m: a dataframe of size (num_levels*num_models, num_components)
    '''
    plt.figure(figsize=(6, 3))
    sns.set(style='white')
    if hue is None:
        sns.violinplot(x=x, y=y, data=importance_df_m)
    else:
        sns.violinplot(x=x, y=y, hue=hue, data=importance_df_m)
    plt.xticks(rotation=45, fontsize=xtick_fontsize)
    plt.yticks(fontsize=ytick_fontsize)
    plt.xlabel(xlab,fontsize=20)
    plt.ylabel('Importance score', fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)




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
importance_df_dict_protocol, time_dict_a_level_dict_protocol = get_importance_all_levels_dict(y_protocol, factor_scores) 
importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = get_importance_all_levels_dict(y_cell_line, factor_scores) 
########### Comparing model run times
time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
########## Comparing time differences between models
time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
plot_runtime_barplot(time_df)

########## Comparing factor scores between models
#### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
importance_df_m = get_melted_importance_df(importance_df_dict)
plot_importance_boxplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison ' + FA_type))
plot_importance_violinplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
plot_importance_boxplot(importance_df_m, x='model', y='importance',title='Model score comparison '+ FA_type)
plot_importance_violinplot(importance_df_m, x='model', y='importance',title='Model score comparison '+ FA_type)


### save importance_df_m to csv
importance_df_m.to_csv('./Results/importance_df_melted_'+
                        'scMixology_'+FA_type+'_'+'baseline.csv')



#### shuffle the covariate vectors n times in a loop
for i in range(n):
    print('i: ', i)

    y_protocol_shuffled = shuffle_covariate(y_protocol)
    y_cell_line_shuffled = shuffle_covariate(y_cell_line)

    ####################################
    #### Importance calculation and run time for model comparison  ######
    ####################################
    importance_df_dict_protocol, time_dict_a_level_dict_protocol = get_importance_all_levels_dict(y_protocol_shuffled, factor_scores) 
    importance_df_dict_cell_line, time_dict_a_level_dict_cell_line = get_importance_all_levels_dict(y_cell_line_shuffled, factor_scores) 
    
    ####################################
    ########### Comparing model run times
    ####################################
    time_df_dict = {**time_dict_a_level_dict_protocol, **time_dict_a_level_dict_cell_line}
    ########## Comparing time differences between models
    time_df = pd.DataFrame.from_dict(time_df_dict, orient='index', columns=list(time_df_dict.values())[0].keys())
    plot_runtime_barplot(time_df)
    

    ############################################################
    ########## Comparing factor scores between models
    ############################################################
    #### merge two importance_df_dict_protocol and importance_df_dict_cell_line dicts
    importance_df_dict = {**importance_df_dict_protocol, **importance_df_dict_cell_line}
    importance_df_m = get_melted_importance_df(importance_df_dict)

    
    plot_importance_boxplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
    plot_importance_violinplot(importance_df_m, x='covariate_level', y='importance', hue='model',
                               title=str('Model score comparison' + FA_type))
    
    plot_importance_boxplot(importance_df_m, x='model', y='importance',title='Model score comparison'+ FA_type)
    plot_importance_violinplot(importance_df_m, x='model', y='importance',title='Model score comparison'+ FA_type)

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


plot_importance_boxplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
plot_importance_violinplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')


############# replace shuffle_0, shuffle_1, ... with shuffle
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].replace(['shuffle_'+str(i) for i in range(n)], 
                                                                              ['shuffle' for i in range(n)])
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].astype('category')
importance_df_m_merged['shuffle'].cat.reorder_categories(['baseline', 'shuffle'], inplace=True)


plot_importance_boxplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
plot_importance_violinplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
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

plot_importance_boxplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
plot_importance_violinplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')

