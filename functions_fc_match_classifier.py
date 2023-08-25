
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


# training classifiers for feature importance on a classification problem


### matching pca factors to different covariates in the data

def get_importance_df(factor_scores, a_binary_cov):

    models = {'LogisticRegression': LogisticRegression(), 
              'DecisionTree': DecisionTreeClassifier(), 'RandomForest': RandomForestClassifier(), 
              'XGB': XGBClassifier(), 'KNeighbors_permute': KNeighborsClassifier()}

    importance_dict = {}
    for model_name, model in models.items():
        X, y = factor_scores, a_binary_cov
        model.fit(X, y)

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
                                           columns=['F'+str(i) for i in range(1, const.num_components+1)])
    return importance_df



#### calculate the mean importance of one level of a given covariate and returns a vector of length const.num_components
def get_mean_importance_level(importance_df_a_level):
    importance_df_np = np.asarray(importance_df_a_level)
    ### scale each row of the importance_df_np to be positive
    importance_df_np = importance_df_np - importance_df_np.min(axis=1, keepdims=True)
    ### normalize each row of the importance_df_np to be between 0 and 1
    importance_df_np = importance_df_np / importance_df_np.max(axis=1, keepdims=True)
    ### calculate the mean of each column of the importance_df_np
    mean_importance = np.mean(importance_df_np, axis=0)
    return mean_importance




#### calculate the mean importance of all levels of a given covariate and returns a dataframe of size (num_levels, const.num_components)
def get_mean_importance_all_levels(covariate_vec, factor_scores):

    mean_importance_df = pd.DataFrame(columns=['PC'+str(i) for i in range(1, const.num_components+1)])

    for covariate_level in np.unique(covariate_vec):
        print('covariate_level: ', covariate_level)

        a_binary_cov = fproc.get_binary_covariate(covariate_vec, covariate_level)
        importance_df_a_level = get_importance_df(factor_scores, a_binary_cov)
        mean_importance_a_level = get_mean_importance_level(importance_df_a_level)

        print('mean_importance_a_level:', mean_importance_a_level)
        mean_importance_df.loc[covariate_level] = mean_importance_a_level

    return mean_importance_df
