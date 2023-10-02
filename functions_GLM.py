import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
from statsmodels import graphics
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import abline_plot


def fit_poisson_GLM(y, x, num_vars) -> dict:
    '''
    fit a Poisson GLM model to each gene of the data
    y: gene expression matrix, cells x genes
    x: design matrix
    num_vars: number of variables in the design matrix
    '''
    

    num_cells = y.shape[0]
    num_genes = y.shape[1]

    ### make an empty array to store the p-values and coefficients
    pvalue = []
    coefficient = []
    yhat = []
    tvalues = []
    resid_pearson = []
    resid_deviance = []
    resid_response = []
    resid_working = []
    fittedvalues = []
    nobs = []
    models = []

    pearson_chi2 = []
    deviance = []
    null_deviance = []

    ### time the fitting process
    start_time = time.time()

    for i in range(len(y[0])):
        y_a_gene = y[:, i]
        model = sm.GLM(y_a_gene, x, family=sm.families.Poisson())
        result = model.fit()
        #print(result.summary())
        
        models.append([result])
        coefficient.append([result.params])
        pvalue.append([result.pvalues]) ## yhat == fittedvalue == mu
        yhat.append([result.predict()])
        fittedvalues.append([result.fittedvalues])

        nobs.append([result.nobs])
        tvalues.append([result.tvalues])
        resid_pearson.append([result.resid_pearson])
        resid_deviance.append([result.resid_deviance])
        resid_response.append([result.resid_response])
        resid_working.append([result.resid_working])
        
        pearson_chi2.append([result.pearson_chi2])
        deviance.append([result.deviance])
        null_deviance.append([result.null_deviance])

    end_time = time.time()
    print('time to fit the model: ', end_time - start_time)

    pvalue = np.asarray(pvalue).reshape(num_genes, num_vars)
    coefficient = np.asarray(coefficient).reshape(num_genes, num_vars)
    tvalues = np.asarray(tvalues).reshape(num_genes, num_vars)

    yhat = np.asarray(yhat).reshape(num_genes, num_cells)
    fittedvalues = np.asarray(fittedvalues).reshape(num_genes, num_cells)
    resid_pearson = np.asarray(resid_pearson).reshape(num_genes, num_cells)
    resid_deviance = np.asarray(resid_deviance).reshape(num_genes, num_cells)
    resid_response = np.asarray(resid_response).reshape(num_genes, num_cells)
    resid_working = np.asarray(resid_working).reshape(num_genes, num_cells)
    nobs = np.asarray(nobs).reshape(num_genes, 1)

    pearson_chi2 = np.asarray(pearson_chi2).reshape(num_genes, 1)
    deviance = np.asarray(deviance).reshape(num_genes, 1)
    null_deviance = np.asarray(null_deviance).reshape(num_genes, 1)

    glm_fit_dict = {'coefficient': coefficient, 'pvalue': pvalue, 'tvalues': tvalues, 'yhat': yhat, 
                    'fittedvalues': fittedvalues, 'resid_pearson': resid_pearson, 'resid_deviance': resid_deviance, 
                    'resid_response': resid_response, 'resid_working': resid_working, 'nobs': nobs, 
                    'pearson_chi2': pearson_chi2, 'deviance': deviance, 'null_deviance': null_deviance}

    return glm_fit_dict



# glm_fit_dict = fit_poisson_GLM(y, x, num_vars)


def plot_glm_diagnostics(glm_fit_dict, y, gene_idx=50):
    '''
    plot the diagnostic plots of the GLM model
    glm_fit_dict: dictionary of the results of the GLM model
    gene_idx: index of the gene to plot
    '''
    ### plot the diagnostic plots of the model outputs
    fittedvalues = glm_fit_dict['fittedvalues']
    y_i = y[:, gene_idx]

    plt.rcParams.update({'font.size': 10})

    fig, ax = plt.subplots()
    ax.scatter(fittedvalues[gene_idx], y[:, gene_idx])
    line_fit = sm.OLS(y_i, sm.add_constant(fittedvalues[gene_idx], prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=ax, alpha=0.5)

    ax.set_title('Model Fit Plot')
    ax.set_ylabel('Observed values')
    ax.set_xlabel('Fitted values')

    sns.jointplot(fittedvalues[gene_idx], y[:, gene_idx], kind='scatter', stat_func=None, color='b', height=4)
    sns.set_context(font_scale=0.9)                                                  
    plt.title('Model Fit Plot')
    plt.xlabel('Fitted values')
    plt.ylabel('Observed values')
    ## change the size of the text in plot
    plt.rc('font', size=10)

    fig, ax = plt.subplots()
    resid_pearson_i = glm_fit_dict['resid_pearson'][gene_idx]
    ax.scatter(fittedvalues[gene_idx], resid_pearson_i, alpha=0.1)
    #ax.hlines(0, 0, 7)
    #ax.set_xlim(0, 7)
    ax.set_title('Residual Dependence Plot')
    ax.set_ylabel('Pearson Residuals')
    ax.set_xlabel('Fitted values')

    sns.jointplot(fittedvalues[gene_idx], resid_pearson_i, kind='scatter', stat_func=None, color='b', height=4)
    sns.set_context(font_scale=0.9)                                                  
    plt.title('Model Fit Plot')
    plt.xlabel('Fitted values')
    plt.ylabel('Pearson Residuals')
    ## change the size of the text in plot
    plt.rc('font', size=10)
    plt.show()

    fig, ax = plt.subplots()
    resid = glm_fit_dict['resid_deviance'][gene_idx]
    resid_std = stats.zscore(resid)
    ax.hist(resid_std, bins=25)
    ax.set_title('Histogram of standardized deviance residuals')

    graphics.gofplots.qqplot(resid, line='r')




def check_null_deviance(glm_fit_res):
    '''
    check if the null deviance is calculated correctly
    glm_fit_res: result of the GLM model

    NullDeviance = 2*(LL(SaturatedModel)−LL(NullModel))
    The value of the deviance function for the model fit with a constant as the only regressor
    GLMResults.llf: Value of the loglikelihood function evalued at params. 
    See statsmodels.families.family for distribution-specific loglikelihoods.

    '''
    null_deviance = 2*(glm_fit_res.llnull - glm_fit_res.llf)

    return np.array_equal(null_deviance, glm_fit_res.null_deviance)



def check_response_residual(glm_fit_res):
    '''
    check if the response residual is calculated correctly
    glm_fit_res: result of the GLM model

    response residual: endog - fittedvalues
    '''
    ## endog: The endogenous response variable: yi
    response_residual = glm_fit_res.endog - glm_fit_res.fittedvalues

    return np.array_equal(response_residual, glm_fit_res.resid_response)




def check_working_residual(glm_fit_res):
    '''
    check if the working residual is calculated correctly
    glm_fit_res: result of the GLM model

    working residual: resid_response/link’(mu)
    '''
    ## Working residual - response residuals scaled by the derivative of the inverse of the link function
    # working residuals = resid_response/link’(mu)

    ## link: The link function of the model
    ## link’(mu): derivative of the link function
    ## mu: The inverse of the link function at the linear predicted values. - then why is mu=yhat instead of mu=e^yhat? 
    ## https://www.statsmodels.org/dev/glm.html#link-functions
    working_residual = glm_fit_res.resid_response/glm_fit_res.family.link.deriv(glm_fit_res.mu)

    return np.array_equal(working_residual, glm_fit_res.resid_working)
  


def check_pearson_residual(glm_fit_res):
    '''
    check if the pearson residual is calculated correctly
    glm_fit_res: result of the GLM model

    pearson residual: (endog - mu)/sqrt(VAR(mu)) where VAR is the distribution specific variance function
    '''
    ## Pearson residuals - response residuals scaled by the square root of the variance function
    # (endog - mu)/sqrt(VAR(mu)) where VAR is the distribution specific variance function
    pearson_residual = (glm_fit_res.endog - glm_fit_res.mu)/np.sqrt(glm_fit_res.family.variance(glm_fit_res.mu))

    return np.array_equal(pearson_residual, glm_fit_res.resid_pearson)


def check_chi2_deviance(glm_fit_res):
    '''
    check if the chi2 deviance is calculated correctly
    glm_fit_res: result of the GLM model

    chi2 deviance: sum of the squares of the Pearson residuals
    '''
    ## Pearson residuals - response residuals scaled by the square root of the variance function
    # (endog - mu)/sqrt(VAR(mu)) where VAR is the distribution specific variance function
    pearson_residual = (glm_fit_res.endog - glm_fit_res.mu)/np.sqrt(glm_fit_res.family.variance(glm_fit_res.mu))

    ## chi2 deviance: sum of the squares of the Pearson residuals
    chi2_deviance = np.sum(pearson_residual**2)

    return np.array_equal(chi2_deviance, glm_fit_res.pearson_chi2)


def check_bic(glm_fit_res):
    '''
    check if the BIC is calculated correctly
    glm_fit_res: result of the GLM model

    BIC: deviance - df_resid * log(nobs)
    '''
    ## BIC : deviance - df_resid * log(nobs)
    BIC = glm_fit_res.deviance - glm_fit_res.df_resid * np.log(glm_fit_res.nobs)

    return np.array_equal(BIC, glm_fit_res.bic)


def check_aic(glm_fit_res):
    '''
    check if the AIC is calculated correctly
    glm_fit_res: result of the GLM model
    
    AIC: aike Information Criterion
    AIC: deviance + 2 * (df_model + 1)
    '''
    ## AIC : deviance + 2 * (df_model + 1)
    ## df_model: rank of the regression matrix excluding the intercept:  df_model = k_exog - 1 = 0 is only strain is included

    AIC = glm_fit_res.deviance + 2 * (glm_fit_res.df_model + 1)

    return np.array_equal(AIC, glm_fit_res.aic)



def check_pseudo_r2(glm_fit_res):
    '''
    check if the pseudo R-squared is calculated correctly
    glm_fit_res: result of the GLM model

    pseudo R-squared: 1 - deviance/null_deviance
    '''
    ## pseudo R-squared: 1 - deviance/null_deviance
    pseudo_r2 = 1 - glm_fit_res.deviance/glm_fit_res.null_deviance

    return np.array_equal(pseudo_r2, glm_fit_res.prsquared)


def save_glm_results(glm_fit_dict, folder_name='GLM_FA_res/'):
    '''
    save the results of the GLM model to csv files
    glm_fit_dict: dictionary of the results of the GLM model
    folder_name: name of the folder to save the results
    '''
    variable_names = glm_fit_dict.keys()
    variable_lists = glm_fit_dict.values()
    for i in range(len(variable_names)):
        np.savetxt( folder_name + variable_names[i] + ".csv", variable_lists[i], delimiter=",")
        print(variable_names[i] + ' saved')



def read_glm_results(folder_name='GLM_FA_res/', 
                     variables_to_read=['coefficient', 'pvalue', 'fittedvalues', 'deviance', 'null_deviance']):
    '''
    read the results of the GLM model from csv files and save them to a dictionary
    folder_name: name of the folder to save the results
    variables_to_read: list of the variables to read
    '''
    ### read the csv file in a loop and save the results to a dictionary
    variable_dict = {}
    for i in range(len(variables_to_read)):
        variable_dict[variables_to_read[i]] = pd.read_csv(folder_name + variables_to_read[i] + '.csv', header=None)
        print(variables_to_read[i] + ' read')
    return variable_dict