import numpy as np
import functions_plotting as fplot
from scipy.special import erf
from itertools import chain
import functions_fc_match_classifier as fmatch 
import pandas as pd
import functions_metrics as fmet
import matplotlib.pyplot as plt

def simulate_gaussian(n,mu, sigma) -> list:
    '''
    simulate n random numbers from normal distribution with mean mu and std sigma
    n: number of samples
    mu: mean
    sigma: standard deviation
    '''
    normal_numbers = []
    for _ in range(n):
        normal_numbers.append(np.random.normal(mu, sigma))
    return normal_numbers



def simulate_mixture_gaussian(n, mu_list, sigma_list, p_list=None) -> list:
    '''
    simulate n random numbers from mixture of normal distributions with mean mu, std sigma, and proportion p
    return a list of the simulated numbers - list of lists
    n: total number of samples
    mu_list: list of means
    sigma_list: list of standard deviations
    p_list: list of proportions of the mixture
    '''
    if p_list is None:
        p_list = [1/len(mu_list)]*len(mu_list)
    mixture_normal_numbers = []
    for i in range(len(mu_list)):
        mixture_normal_numbers.append(simulate_gaussian(round(n*p_list[i]), mu_list[i], sigma_list[i]))

    return mixture_normal_numbers



def calc_overlap_double_Gaussian(mu1, mu2, sigma1, sigma2) -> float:
    '''
    calculate the overlap between two normal distributions with mean mu, std sigma, and proportion p
    m1: mean of the first normal distribution
    m2: mean of the second normal distribution
    sigma1: standard deviation of the first normal distribution
    sigma2: standard deviation of the second normal distribution
    '''
    # calculate the overlap between two normal distributions
    # https://stats.stackexchange.com/questions/103800/calculate-probability-area-under-the-overlapping-area-of-two-normal-distributi
    
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1

    if sigma1 == sigma2:
        c = (mu1 + mu2)/2

    elif sigma1 != sigma2:
        c_numerator = mu2*sigma1**2 - sigma2*(mu1*sigma2 + sigma1*np.sqrt((mu1-mu2)**2 + 2*(sigma1**2 - sigma2**2)*np.log(sigma1/sigma2)))
        c_denominator = sigma1**2 - sigma2**2
        c = c_numerator/c_denominator

    print('c (pdf intersection): ', c)
    # P(X_1>c) + P(X_2<c) = 1 - F_1(c) + F_2(c)
    # 1 - 0.5*erf((c-mu1)/(sqrt(2)*sigma1)) + 0.5*erf((c-mu2)/(sqrt(2)*sigma2))
    overlap = 1 - 0.5*erf((c-mu1)/(np.sqrt(2)*sigma1)) + 0.5*erf((c-mu2)/(np.sqrt(2)*sigma2))
    return overlap



def get_random_number(min=0, max=1, round_digit=3) -> float:
    '''
    generate a random number from uniform distribution between min and max
    min: minimum value
    max: maximum value
    round_digit: number of digits to round to
    '''
    return round(np.random.uniform(min,max),round_digit)



def get_random_list(n, min, max) -> list:
    '''
    returns a list of n random numbers from uniform distribution between 0 and 1
    n: length of the list
    min: minimum value
    max: maximum value
    '''
    random_list = []
    for _ in range(n):
        random_list.append(get_random_number(min=min, max=max))
    return random_list
    


def get_random_list_sum_1(n) -> list:
    '''
    returns a list of n random numbers from uniform distribution between min and max
    n: length of the list
    min: minimum value
    max: maximum value
    '''
    random_list = [get_random_number(min=0, max=10) for _ in range(n)]
    s = sum(random_list)
    random_list = [ i/s for i in random_list ]

    return random_list



def get_random_factor_parameters(num_mixtures=2, mu_min=0, mu_max=None,
                      sigma_min=0.5, sigma_max=1,
                      p_equals=True) -> tuple:
    '''
    num_mixtures: number of mixtures
    mu_min: minimum value of the mean
    mu_max: maximum value of the mean
    sigma_min: minimum value of the standard deviation
    sigma_max: maximum value of the standard deviation
    p_equals: if True, all the mixtures have equal proportions

    '''
    if mu_max is None:
        mu_max = num_mixtures*2

    mu_list = get_random_list(num_mixtures, min=mu_min, max=mu_max)
    sigma_list = get_random_list(num_mixtures, min=sigma_min, max=sigma_max)

    if p_equals:
        p_list = [1/num_mixtures]*num_mixtures
    else:
        p_list = get_random_list_sum_1(num_mixtures)

    return mu_list, sigma_list, p_list



def get_a_factor_pairwise_overlap(mu_list, sigma_list) -> np.array:
    '''
    calculate the pairwise overlap between all the normal distributions in a mixture
    mu_list: list of means
    sigma_list: list of standard deviations
    '''
    
    overlap_matrix = np.zeros((len(mu_list),len(mu_list)))
    #overlap_matrix.fill(-1)

    for i in range(len(mu_list)):
        for j in range(len(mu_list)):
            overlap_matrix[i,j] = calc_overlap_double_Gaussian(mu1=mu_list[i], mu2=mu_list[j], 
                                                               sigma1=sigma_list[i], sigma2=sigma_list[j])
            print('mu',i,':' ,mu_list[i], ' sigma', i,':', sigma_list[i] ,
                   ' and  mu',j, ':' ,mu_list[j],' sigma',j,':', sigma_list[j], 
             ' - overlap: ',overlap_matrix[i,j])
            
    return overlap_matrix


def get_simulated_factor_object(n=10000,num_mixtures=2, mu_min=0, mu_max=None,
                      sigma_min=0.5, sigma_max=1,
                      p_equals=True) -> tuple:
    '''
    num_mixtures: number of mixtures
    mu_min: minimum value of the mean
    mu_max: maximum value of the mean
    sigma_min: minimum value of the standard deviation
    sigma_max: maximum value of the standard deviation
    p_equals: if True, all the mixtures have equal proportions

    '''
    mu_list, sigma_list, p_list = get_random_factor_parameters(num_mixtures=num_mixtures, mu_min=mu_min, mu_max=mu_max,
                      sigma_min=sigma_min, sigma_max=sigma_max, p_equals=p_equals)
    a_random_factor = simulate_mixture_gaussian(n=n, mu_list=mu_list, sigma_list=sigma_list, p_list=p_list)
    overlap_matrix = get_a_factor_pairwise_overlap(mu_list, sigma_list)
    
    return a_random_factor, overlap_matrix, mu_list, sigma_list, p_list



def get_sim_factor_covariates(a_random_factor):
    '''
    make a list of covariates for each factor
    a_random_factor: a list of lists of simulated numbers
    '''
    a_factor_covariate_list = []
    for i in range(len(a_random_factor)):
        for _ in range(len(a_random_factor[i])):
            a_factor_covariate_list.append('cov'+ str(i+1))

    return a_factor_covariate_list



def get_covariate_freq_table(covariate_list):
    '''
    make a frequency table of covariate_list elements
    covariate_list: list of covariates
    '''
    covariate_freq = {}
    for i in covariate_list:
        if i in covariate_freq:
            covariate_freq[i] += 1
        else:
            covariate_freq[i] = 1
    return covariate_freq


def unlist(l):
    '''
    unlist a list of lists
    '''
    return [item for sublist in l for item in sublist]


def get_pairwise_match_score_matrix(match_score_df, factor_index):
    '''
    make a matrix of pairwise match scores between all covs for a given factor
    match_score_df: dataframe of match scores (covariates, factors)
    factor_index: index of the factor
    '''

    ### make a matrix of pairwise match scores between all covs for a given factor
    match_score_matrix = np.zeros((match_score_df.shape[0],match_score_df.shape[0]))

    for i in range(match_score_df.shape[0]):
        for j in range(match_score_df.shape[0]):
            match_score_f_covi = AUC_all_factors_df.iloc[i,factor_index]
            match_score_f_covj = AUC_all_factors_df.iloc[j,factor_index]
            pairwise_match_score = np.sqrt(match_score_f_covi*match_score_f_covj)
            match_score_matrix[i,j] = pairwise_match_score

    return match_score_matrix



def convert_matrix_list_to_vector(matrix_list):
    '''
    convert a list of matrices to a flat vector - remove the upper triangle and the diagonal
    matrix_list: list of matrices
    '''
    vector_list = []
    for i in range(len(matrix_list)):
        mat_tri = mask_upper_triangle(matrix_list[i])
        mat_tri_flat = mat_tri.flatten()
        ## remove the nan values
        mat_tri_flat = mat_tri_flat[~np.isnan(mat_tri_flat)]
        vector_list.append(mat_tri_flat)
    
    vector_list = unlist(vector_list)

    return vector_list


### define a function to mask the upper triangle of a matrix
def mask_upper_triangle(mat):
    '''
    mask the upper triangle of a matrix
    '''
    mat_triu = np.triu(mat).T
    mat_triu[mat_triu == 0] = np.nan
    ## mask the diagonal
    np.fill_diagonal(mat_triu, np.nan)

    return mat_triu


def plot_scatter(overlap_scores, matching_score, title=''):
    '''
    plot the scatter plot of the overlap and match scores
    overlap_scores: vector of overlap scores
    matching_score: vector of matching scores
    '''
    ### use matplotlib to plot the scatter plot
    
    plt.scatter(overlap_scores, matching_score)
    ### fit a line to the scatter plot
    plt.plot(np.unique(overlap_scores), np.poly1d(np.polyfit(overlap_scores, matching_score, 1))(np.unique(overlap_scores)))
    ### add value of correlation coefficient
    plt.text(0.5, 0.5, 'R: '+ str(round(np.corrcoef(overlap_scores, matching_score)[0,1], 2)))
    plt.xlabel('overlap')
    plt.ylabel('match score')
    plt.title(title)
    plt.show()

### TODO: Define a class for a factor_sim
### factor has the following attributes:
### - num_mixtures, mu_list, sigma_list, p_list, overlap_matrix


num_factors = 10
sim_factors_list = []
overlap_mat_list = []
covariate_list = []
num_mixtures = 5 ## each factor is a mixture of 3 normal distributions

for i in range(num_factors):
    a_random_factor, overlap_matrix, mu_list, sigma_list, p_list = get_simulated_factor_object(n=10000,num_mixtures=num_mixtures, 
                                                                                               mu_min=0, mu_max=None,
                                                                                               sigma_min=0.5, sigma_max=1, p_equals=True)  
    sim_factors_list.append(unlist(a_random_factor))
    overlap_mat_list.append(overlap_matrix)
    covariate_list.append(get_sim_factor_covariates(a_random_factor))

len(sim_factors_list)
len(sim_factors_list[0])


### convert sim_factors_list to a numpy nd array with shape (num_samples, num_factors)
sim_factors_array = np.asarray(sim_factors_list).T
sim_factors_array.shape
sim_factors_df = pd.DataFrame(sim_factors_array, columns=['factor'+str(i+1) for i in range(num_factors)])
sim_factors_df

### convert covariate_list[0] to pandas.core.series.Series
covariate_vector = pd.Series(covariate_list[0])
factor_scores = sim_factors_array

####################################
#### Matching between factors and covariates ######
####################################

### calculate the mean importance of each covariate level
mean_importance_df = fmatch.get_mean_importance_all_levels(covariate_vector, factor_scores)
fplot.plot_all_factors_levels_df(mean_importance_df, title='F-C Match: Feature importance scores', color='coolwarm')
all_covariate_levels = mean_importance_df.index.values

#### AUC score
#### calculate the AUC of all the factors for all the covariate levels
AUC_all_factors_df, wilcoxon_pvalue_all_factors_df = fmet.get_AUC_all_factors_df(factor_scores, covariate_vector)
fplot.plot_all_factors_levels_df(AUC_all_factors_df, 
                                 title='F-C Match: AUC scores', color='YlOrBr')

#################################### 


fplot.plot_histogram(a_random_factor, 'normal distribution')
for i in range(num_mixtures):
    fplot.plot_histogram(a_random_factor[i], 'normal distribution')


factor_index = 0
match_score_mat_AUC = get_pairwise_match_score_matrix(AUC_all_factors_df,factor_index)
match_score_mat_meanImp = get_pairwise_match_score_matrix(mean_importance_df,factor_index)
overlap_mat = overlap_mat_list[factor_index]

### plot the scatter plot of the overlap and match scores
plot_scatter(overlap_mat.flatten(), match_score_mat_AUC.flatten(), title='AUC')
plot_scatter(overlap_mat.flatten(), match_score_mat_meanImp.flatten(), title='feature importance')

#### calculating the scores for all the factors
match_score_mat_AUC_list = []
match_score_mat_meanImp_list = []

for i in range(num_factors): ## i is the factor index
    match_score_mat_AUC_list.append(get_pairwise_match_score_matrix(AUC_all_factors_df,i))
    match_score_mat_meanImp_list.append(get_pairwise_match_score_matrix(mean_importance_df,i))

match_score_mat_AUC_flat = convert_matrix_list_to_vector(match_score_mat_AUC_list)
match_score_mat_meanImp_flat = convert_matrix_list_to_vector(match_score_mat_meanImp_list)
overlap_mat_flat = convert_matrix_list_to_vector(overlap_mat_list)

plot_scatter(overlap_mat_flat, match_score_mat_meanImp_flat, title='feature importance')
plot_scatter(overlap_mat_flat, match_score_mat_AUC_flat, title='AUC')
