
import sys
sys.path.append('./Code/')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

import functions_processing as fproc
import constants as const

import skimage as ski
import scipy.stats as ss

# training classifiers for feature importance on a classification problem
# matching pca factors to different covariates in the data

def get_importance_df(factor_scores, a_binary_cov, time_eff=True) -> pd.DataFrame:
    '''
    calculate the importance of each factor for each covariate level
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    a_binary_cov: numpy array of the binary covariate for a covariate level (n_cells, )
    time_eff: if True, skip RandomForest which is time consuming
    '''

    models = {'LogisticRegression': LogisticRegression(), 
              'DecisionTree': DecisionTreeClassifier(), 
              'XGB': XGBClassifier(), 'KNeighbors_permute': KNeighborsClassifier()}
    
    if not time_eff:
        models['RandomForest'] = RandomForestClassifier()

    importance_dict = {}
    for model_name, model in models.items():
        X, y = factor_scores, a_binary_cov
        model.fit(X, y)

        if model_name == 'LogisticRegression':
            ### use the absolute value of the logistic reg coefficients as the importance - for consistency with other classifiers
            importance_dict[model_name] = np.abs(model.coef_)[0]
            #importance_dict[model_name] = model.coef_[0]

        elif model_name in ['DecisionTree', 'RandomForest', 'XGB']:
            # get importance
            importance_dict[model_name] = model.feature_importances_
        else:
            # perform permutation importance
            results = permutation_importance(model, X, y, scoring='accuracy')
            importance_dict[model_name] = np.abs(results.importances_mean)

    importance_df = pd.DataFrame.from_dict(importance_dict, orient='index', 
                                           columns=['F'+str(i) for i in range(1, factor_scores.shape[1]+1)])
    return importance_df



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



def get_mean_importance_all_levels(covariate_vec, factor_scores, scale='rank', mean='geometric') -> pd.DataFrame:
    '''
    calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, num_components)
    covariate_vec: numpy array of the covariate vector (n_cells, )
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''


    mean_importance_df = pd.DataFrame(columns=['F'+str(i) for i in range(1, factor_scores.shape[1]+1)])

    for covariate_level in np.unique(covariate_vec):
        print('covariate_level: ', covariate_level)

        a_binary_cov = fproc.get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level = get_importance_df(factor_scores, a_binary_cov)
        mean_importance_a_level = get_mean_importance_level(importance_df_a_level, scale, mean)

        print('mean_importance_a_level:', mean_importance_a_level)
        mean_importance_df.loc[covariate_level] = mean_importance_a_level

    return mean_importance_df



def get_percent_matched_factors(mean_importance_df, threshold) -> float:
      total_num_factors = mean_importance_df.shape[1]
      matched_factor_dist = np.sum(mean_importance_df > threshold)

      num_matched_factors = np.sum(matched_factor_dist>0)
      percent_matched = np.round((num_matched_factors/total_num_factors)*100, 2)
      return matched_factor_dist, percent_matched


def get_percent_matched_covariate(mean_importance_df, threshold) -> float:
      total_num_covariates = mean_importance_df.shape[0]
      matched_covariate_dist = np.sum(mean_importance_df > threshold, axis=1)

      num_matched_cov = np.sum(matched_covariate_dist>0)
      percent_matched = np.round((num_matched_cov/total_num_covariates)*100, 2)
      return matched_covariate_dist, percent_matched



def get_otsu_threshold(values) -> float:
      '''
      This function calculates the otsu threshold of the feature importance scores
      :param values: a 1D array of values
      :return: threshold
      '''
      threshold = ski.filters.threshold_otsu(values)
      return threshold