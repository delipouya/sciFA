import numpy as np
import pandas as pd
import sklearn.cluster as cluster
from sklearn import metrics
import scipy as sp

import functions_processing as fproc
import constants as const

### calculating specificity of a single factor
def get_factor_specificity(factor_i, mean_importance_df, p_all_factors) -> float:
    '''
    calculate the specificity of a factor based on the mean importance matrix
    factor_i: the index of the factor
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    p_all_factors: numpy array of probability of a factor being selected
    '''
    p_factor_i = p_all_factors[factor_i]
    S_factor = 0 
    for lev in range(mean_importance_df.shape[0]):
        p_factor_lev = mean_importance_df.iloc[lev, factor_i]
        S_factor += (p_factor_lev/p_factor_i)*np.log(p_factor_lev/p_factor_i)
    return S_factor/mean_importance_df.shape[0]


def get_all_factors_specificity(mean_importance_df) -> list:
    '''
    calculate the specificity of all the factors based on the mean importance matrix
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    '''
    factor_specificity_all = []
    p_all_factors = np.sum(np.asarray(mean_importance_df), axis=0)/mean_importance_df.shape[0]
    for factor_i in range(mean_importance_df.shape[1]):
        S_factor = get_factor_specificity(factor_i, mean_importance_df, p_all_factors)
        factor_specificity_all.append(S_factor)
    return factor_specificity_all



def get_factor_entropy(a_factor) -> float:
    '''
    calculate the entropy of a factor
    a_factor: numpy array of the factor values
    '''
    ### caculate number of zeros in pk
    num_zeros = np.count_nonzero(a_factor == 0)
    #print('num_zeros: ', num_zeros)
    ### shift all the values to be positive
    a_factor = a_factor - np.min(a_factor) + 1e-10
    ### devide all the score by the max value
    a_factor = a_factor / np.max(a_factor)
    #plt.hist(a_factor, bins=100)
    H = -sum(a_factor * np.log(a_factor))
    return H


def get_factor_entropy_all(factor_scores) -> list:
    ''' calculate the entropy of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    H_all = []
    for i in range(const.num_components):
        a_factor = factor_scores[:,i]
        H = get_factor_entropy(a_factor)
        H_all.append(H)
    return H_all


def get_factor_variance_all(factor_scores) -> list:
    ''' calculate the variance of all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    factor_variance_all = []
    for i in range(const.num_components):
        a_factor = factor_scores[:,i]
        factor_variance = np.var(a_factor)
        factor_variance_all.append(factor_variance)
    return factor_variance_all


def get_scaled_variance_level(a_factor, covariate_vector, covariate_level) -> float:
    ''' 
    calculate the scaled variance of one factor and one covariate
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    covariate_level: one level of interest in the covariate
    '''
    ### select the cells in a_factor that belong to the covariate level
    a_factor_subset = a_factor[covariate_vector == covariate_level] 
    ### scaled variance of a factor and a covariate level
    scaled_variance = np.var(a_factor_subset)/np.var(a_factor) 
    return scaled_variance


def get_SV_all_levels(a_factor, covariate_vector) -> list:
    '''
    calculate the scaled variance for all the levels in a covariate
    represents how well mixed the factor is across each level of the covariate. 
    scaled_variance = 1 is well mixed, 0 is not well mixed
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    '''
    scaled_variance_all = []
    for covariate_level in covariate_vector.unique():
        scaled_variance = get_scaled_variance_level(a_factor, covariate_vector, covariate_level)
        scaled_variance_all.append(scaled_variance)
        
    return scaled_variance_all


def get_a_factor_ASV(a_factor, covariate_vector, mean_type='geometric') -> float:
    '''
    calculate an average for the relative scaled variance for all the levels in a covariate
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    mean_type: the type of mean to calculate the average scaled variance
    '''

    ### calculate the relative scaled variance for all the levels in a covariate
    scaled_variance_all = get_SV_all_levels(a_factor, covariate_vector)
    print('mean type: ', mean_type)
    ### calculate the geometric mean of the scaled variance for all levels of the covariate
    if mean_type == 'geometric':
        RSV = np.exp(np.mean(np.log(scaled_variance_all)))

    elif mean_type == 'arithmetic':
        ### sum of the scaled variance for all levels of the covariate / number of levels in the covariate
        RSV = sum(scaled_variance_all)/len(np.unique(covariate_vector)) 

    return RSV


def get_ASV_all(factor_scores, covariate_vector, mean_type='geometric') -> list:
    '''
    calculate the average scaled variance for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    mean_type: the type of mean to calculate the average scaled variance
    '''

    ASV_all = []
    for i in range(const.num_components):
        a_factor = factor_scores[:,i]
        ASV = get_a_factor_ASV(a_factor, covariate_vector, mean_type)
        ASV_all.append(ASV)
    return ASV_all


def get_factors_SV_all_levels(factor_scores, covariate_vector) -> np.array:
    '''
    calculate the scaled variance for all the levels in a covariate for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    covariate_vector: numpy array of the covariate values for all the cells (n_cells, 1)
    '''
    SV_all_factors = []
    for i in range(const.num_components):
        a_factor = factor_scores[:,i]
        SV_all = get_SV_all_levels(a_factor, covariate_vector)
        
        SV_all_factors.append(SV_all)
    SV_all_factors = np.asarray(SV_all_factors)
    return SV_all_factors.T ### traspose the matrix to have the factors in columns and cov levels in rows



def get_silhouette_score(a_factor, kmeans_labels) -> float:
    ''' calculate the silhouette score for a factor
    a_factor: numpy array of the one factor scores for all the cells (n_cells, 1)
    kmeans_labels: numpy array of the kmeans labels for all the cells (n_cells, 1)
    '''
    a_factor_silhouette_score = metrics.silhouette_score(a_factor.reshape(-1, 1), kmeans_labels, metric='euclidean')
    return a_factor_silhouette_score



def get_kmeans_silhouette_scores(factor_scores) -> dict:
    ''' calculate the silhouette score for all the factors
    factor_scores: numpy array of the factor scores for all the cells (n_cells, n_factors)
    '''
    silhouette_score_all = []
    kmeans_all = []
    for i in range(const.num_components):
        ### apply kmeans to all the factors independently
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(factor_scores[:,i].reshape(-1, 1))
        a_factor_silhouette_score = get_silhouette_score(factor_scores[:,i], kmeans.labels_)
        silhouette_score_all.append(a_factor_silhouette_score)
        kmeans_all.append(kmeans.labels_)
    return {'silhouette': silhouette_score_all, 'kmeans': kmeans_all}



def get_scaled_metrics(all_metrics_df) -> np.array:
    '''
    Return numpy array of the scaled all_metrics pandas df based on each metric 
    all_metrics_df: a pandas dataframe of the metrics for all the factors
    '''
    all_metrics_np = all_metrics_df.to_numpy()

    all_metrics_scaled = np.concatenate((fproc.get_scaled_vector(all_metrics_np[:,0]).reshape(-1, 1),
                                fproc.get_scaled_vector(all_metrics_np[:,1]).reshape(-1, 1),
                                fproc.get_scaled_vector(all_metrics_np[:,2]).reshape(-1, 1),
                                fproc.get_scaled_vector(all_metrics_np[:,3]).reshape(-1, 1),
                                fproc.get_scaled_vector(all_metrics_np[:,4]).reshape(-1, 1),
                                fproc.get_scaled_vector(all_metrics_np[:,5]).reshape(-1, 1)
                                ),axis=1)

    return all_metrics_scaled





def get_AUC_alevel(a_factor, covariate_vector, covariate_level) -> float:
    '''
    calculate the AUC of a factor for a covariate level
    return the AUC and the p-value of the U test
    a_factor: a factor score
    covariate_vector: a vector of the covariate
    covariate_level: a level of the covariate

    '''
    n1 = np.sum(covariate_vector==covariate_level)
    n0 = len(a_factor)-n1
    
    ### U score manual calculation
    #order = np.argsort(a_factor)
    #rank = np.argsort(order)
    #rank += 1   
    #U1 = np.sum(rank[covariate_vector == covariate_level]) - n1*(n1+1)/2

    ### calculate the U score using scipy
    scipy_U = sp.stats.mannwhitneyu(a_factor[covariate_vector == covariate_level] , 
                                    a_factor[covariate_vector != covariate_level] , 
                                    alternative="two-sided", use_continuity=False)
    
    AUC1 = scipy_U.statistic/ (n1*n0)
    return AUC1, scipy_U.pvalue

def get_AUC_all_levels(a_factor, covariate_vector) -> list:
    '''
    calculate the AUC of a factor for all the covariate levels
    return a list of AUCs for all the covariate levels
    a_factor: a factor score
    covariate_vector: a vector of the covariate
    '''
    AUC_all = []
    for covariate_level in np.unique(covariate_vector):
        AUC1, pvalue = get_AUC_alevel(a_factor, covariate_vector, covariate_level)
        AUC_all.append(AUC1)
    return AUC_all

def get_AUC_all_factors(factor_scores, covariate_vector) -> list:
    '''
    calculate the AUC of all the factors for all the covariate levels
    return a list of AUCs for all the factors
    factor_scores: a matrix of factor scores
    covariate_vector: a vector of the covariate
    '''
    AUC_all_factors = []
    for i in range(const.num_components):
        a_factor = factor_scores[:,i]
        AUC_all = get_AUC_all_levels(a_factor, covariate_vector)
        AUC_all_factors.append(AUC_all)
    return AUC_all_factors

def get_AUC_all_factors_df(factor_scores, covariate_vector) -> pd.DataFrame:
    '''
    calculate the AUC of all the factors for all the covariate levels
    return a dataframe of AUCs for all the factors
    factor_scores: a matrix of factor scores
    covariate_vector: a vector of the covariate
    '''
    AUC_all_factors = get_AUC_all_factors(factor_scores, covariate_vector)
    AUC_all_factors_df = pd.DataFrame(AUC_all_factors).T
    AUC_all_factors_df.columns = ['F'+str(i+1) for i in range(const.num_components)]
    AUC_all_factors_df.index = np.unique(covariate_vector)
    return AUC_all_factors_df

