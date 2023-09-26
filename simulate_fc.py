import numpy as np
import functions_plotting as fplot
from scipy.special import erf

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




num_mixtures = 3
mu_list, sigma_list, p_list = get_random_factor_parameters(num_mixtures=num_mixtures, mu_min=0, mu_max=None,
                      sigma_min=0.5, sigma_max=1, p_equals=True)

a_random_factor = simulate_mixture_gaussian(n=100000, mu_list=mu_list, sigma_list=sigma_list, p_list=p_list)
overlap_matrix = get_a_factor_pairwise_overlap(mu_list, sigma_list)
overlap_matrix


fplot.plot_histogram(a_random_factor, 'normal distribution')
for i in range(num_mixtures):
    fplot.plot_histogram(a_random_factor[i], 'normal distribution')




