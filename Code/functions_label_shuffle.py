
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
import seaborn as sns

import functions_processing as fproc

import numpy as np
import pandas as pd
from scipy.io import mmread
import matplotlib.pyplot as plt

import ssl; ssl._create_default_https_context = ssl._create_unverified_context
import functions_processing as fproc
import functions_fc_match_classifier as fmatch 
import statsmodels.api as sm
import scipy.stats as ss
 

np.random.seed(10)
import time




# training classifiers for feature importance on a classification problem
# matching pca factors to different covariates in the data

def get_importance_df_v_comp(factor_scores, a_binary_cov) -> pd.DataFrame:
    '''
    calculate the importance of each factor for each covariate level - model comparison version
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


        ### get the importance_df for each covariate level
        a_binary_cov = fproc.get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level, time_dict = get_importance_df_v_comp(factor_scores, a_binary_cov)
        ### save the importance_df (each column's a model) for each covariate level
        importance_df_a_level_dict[covariate_level] = importance_df_a_level
        time_dict_a_level_dict[covariate_level] = time_dict


        
    return importance_df_a_level_dict, time_dict_a_level_dict



def get_mean_importance_level(importance_df_a_level, scale, mean) -> np.array:
    ''' 
    calculate the mean importance of one level of a given covariate and returns a vector of length of number of factors
    importance_df_a_level: a dataframe of the importance of each factor for a given covariate level
    scale: 'standard', 'minmax' or 'rank', 'pearson'
    standard: scale each row of the importance_df_np to have zero mean and unit variance "REMOVED"
    minmax: scale each row of the importance_df_np to be between 0 and 1
    rank: replace each row of the importance_df_np with its rank
    mean: 'arithmatic' or 'geometric'
    arithmatic: calculate the arithmatic mean of each column
    geometric: calculate the geometric mean of each column
'
    '''
    importance_df_np = np.asarray(importance_df_a_level)
    ### normalize the importance score of each classifier in importance_df_np matrix
    if scale == 'standard':
        ### scale each row of the importance_df_np to have zero mean and unit variance
        importance_df_np = (importance_df_np - importance_df_np.mean(axis=1, keepdims=True))/importance_df_np.std(axis=1, keepdims=True)
    if scale == 'minmax':
        ### scale each row of the importance_df_np to be between 0 and 1
        importance_df_np = (importance_df_np - importance_df_np.min(axis=1, keepdims=True))/(importance_df_np.max(axis=1, keepdims=True) - importance_df_np.min(axis=1, keepdims=True))
    elif scale == 'rank':
        ### replace each row of the importance_df_np with its rank
        importance_df_np = np.apply_along_axis(ss.rankdata, 1, importance_df_np)

    ### calculate the mean of the importance_df_np matrix
    if mean == 'arithmatic':
        ### calculate the arithmatic mean of each column
        mean_importance = np.mean(importance_df_np, axis=0)
    elif mean == 'geometric':
        ### calculate the geometric mean of each column
        mean_importance = ss.gmean(importance_df_np, axis=0)

    return mean_importance



def get_mean_importance_df_v_comp(importance_df_levels_dict) -> pd.DataFrame:
        '''
        calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, num_components)
        the mean importance is calculated with different scaling and mean calculation methods for comparison
        importance_df_a_level_dict: a dictionary of dataframes of the importance of each factor for a given covariate level
        '''
        
        ### convert dictionary keys to list
        first_cov = list(importance_df_levels_dict.keys())[0]
        num_factors = importance_df_levels_dict[first_cov].shape[1]
        
        mean_imp_standard_arith_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])
        mean_imp_standard_geom_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])

        mean_imp_minmax_arith_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])
        mean_imp_minmax_geom_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])

        mean_imp_rank_arith_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])
        mean_imp_rank_geom_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, num_factors+1)])
        

        ### loop over all covariate levels in importance_df_a_level_dict with key as covariate_level and value as importance_df_a_level
        for covariate_level, importance_df_a_level in importance_df_levels_dict.items():
            print('covariate_level: ', covariate_level)
        
            ### calculate the mean importance of all models for each covariate level
            #### scale: standard - mean: arithmatic
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='standard', mean='arithmatic')
            mean_imp_standard_arith_df.loc[covariate_level] = mean_importance_a_level

            #### scale: standard - mean: geometric
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='standard', mean='geometric')
            mean_imp_standard_geom_df.loc[covariate_level] = mean_importance_a_level

            #### scale: minmax - mean: arithmatic
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='minmax', mean='arithmatic')
            mean_imp_minmax_arith_df.loc[covariate_level] = mean_importance_a_level

            #### scale: minmax - mean: geometric
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='minmax', mean='geometric')
            mean_imp_minmax_geom_df.loc[covariate_level] = mean_importance_a_level

            #### scale: rank - mean: arithmatic
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='rank', mean='arithmatic')
            mean_imp_rank_arith_df.loc[covariate_level] = mean_importance_a_level

            #### scale: rank - mean: geometric
            mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale='rank', mean='geometric')
            mean_imp_rank_geom_df.loc[covariate_level] = mean_importance_a_level


        return mean_imp_standard_arith_df, mean_imp_standard_geom_df, mean_imp_minmax_arith_df, mean_imp_minmax_geom_df, mean_imp_rank_arith_df, mean_imp_rank_geom_df


#importance_df_levels_dict = importance_df_dict_protocol
#meanimp_rank_arith_protocol, meanimp_minmax_arith_protocol, meanimp_rank_geom_protocol, meanimp_minmax_geom_protocol = flabel.get_mean_importance_df_v_comp(importance_df_dict_protocol)


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

