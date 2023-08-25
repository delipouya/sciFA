import numpy as np
import scanpy as sc
import pandas as pd
import statsmodels.api as sm
import constants as const


def import_AnnData(path_to_file) -> sc.AnnData:
    """Import data from a file and return a numpy array.
    path_to_file: path to the file
    """
    #### import the immune subpopulation of the rat samples
    data = sc.read(path_to_file) ## attributes removed
    data.var_names_make_unique()

    ### renaming the meta info column names: https://github.com/theislab/scvelo/issues/255
    data.__dict__['_raw'].__dict__['_var'] = data.__dict__['_raw'].__dict__['_var'].rename(columns={'_index': 'features'})

    return data


def get_metadata_scMix(data) -> tuple:
    """Return the metadata of the scMixology dataset, including cell-line, sample and protocol information.
    data: AnnData object
    """

    #### sample metadata
    y_cell_line = data.obs.cell_line_demuxlet
    y_sample = data.obs[['sample']].squeeze()

    #### adding a column to data object for protocol
    ## empty numpy array in length of the number of cells
    y_protocol = np.empty((data.n_obs), dtype="S10")

    for i in range(data.n_obs):
        if data.obs['sample'][i] in ['sc_10x', 'sc_10X']:
            y_protocol[i] = 'sc_10X'

        elif data.obs['sample'][i] == 'Dropseq':
            y_protocol[i] = 'Dropseq'
            
        else:
            y_protocol[i] = 'CELseq2'

    
    # data.obs['protocol'] = y_protocol
    y_protocol = pd.Series(y_protocol)
    y_protocol.unique()

    return y_cell_line, y_sample, y_protocol
    


def get_data_array(data) -> np.array:
    """Return the data matrix as a numpy array, and the number of cells and genes.
    data: AnnData object
    """

    data_numpy = data.X.toarray()
    
    cell_sums = np.sum(data_numpy,axis=1) # row sums - library size
    gene_sums = np.sum(data_numpy,axis=0) # col sums - sum reads in a gene
    data_numpy = data_numpy[:,gene_sums != 0]


    ## working with the rat data
    num_cells = data_numpy.shape[0]
    num_genes = data_numpy.shape[1]

    print(num_cells, num_genes)

    return data_numpy, num_cells, num_genes



def get_sub_data(y) -> np.array:    
    ''' subset the data matrix to the top num_genes genes
    y: numpy array of the gene expression matrix (n_cells, n_genes)
    '''

    #### select num_genes genes based on variance
    ### calculate the variance for each gene
    gene_vars = np.var(y, axis=0)
    ### select the top num_genes genes with the highest variance
    gene_idx = np.argsort(gene_vars)[::-1][0:const.num_genes]

    #### randomly select num_genes genes
    #gene_idx = random.sample(range(0, y.shape[1]), num_genes)

    ### subset the data matrix to the top num_genes genes
    return y[:, gene_idx]


def get_binary_covariate_v1(covariate, covariate_level, data) -> np.array:
    ''' return a binary covariate vector for a given covariate and covariate level
    covariate: a column of the dat object metadata
    covariate_level: one level of the covariate
    data: AnnData object
    '''
    covariate_list = np.zeros((data.obs.shape[0]))
    for i in range(data.obs.shape[0]):
        ### select the ith element of 
        if data.obs[[covariate]].squeeze()[i] == covariate_level:
            covariate_list[i] = 1
    return covariate_list


def get_binary_covariate(covariate_vec, covariate_level) -> np.array:
    ''' return a binary covariate vector for a given covariate and covariate level
    covariate_vec: a vector of values for a covariate
    covariate_level: one level of the covariate
    '''
    covariate_list = np.zeros((len(covariate_vec)))
    for i in range(len(covariate_vec)):
        ### select the ith element of 
        if covariate_vec[i] == covariate_level:
            covariate_list[i] = 1
    return covariate_list


def get_design_mat(a_metadata_col, data) -> np.array:
    ''' return a onehot encoded design matrix for a given column of the dat object metadata
    a_metadata_col: a column of the dat object metadata
    data: AnnData object
    '''
    
    column_levels = data.obs[a_metadata_col].unique()
    dict_covariate = {}
    for column_level in column_levels:
        print(column_level)
        dict_covariate[column_level] = get_binary_covariate(a_metadata_col, column_level, data)

    #### stack colummns of dict_covariate 
    x = np.column_stack((dict_covariate[column] for column in column_levels))
    return x



def get_protocol_designmat_scMixology(data) -> np.array:
    ''' return a design matrix for the protocol covariate
    data: AnnData object
    '''

    #### Design matrix : Intercept + Depth + protocol
    x_protocol = get_design_mat('protocol', data)
    # x_cell_line = get_design_mat('cell_line', data)
    x = np.column_stack((data.obs.nCount_originalexp, x_protocol)) #, x_batch
    x = sm.add_constant(x) ## adding the intercept

    return x


def get_lib_designmat_scMixology(data):
    ''' return a design matrix for the library size covariate
    data: AnnData object
    '''
    x = np.column_stack((np.ones(data.nobs), np.array(data.obs.nCount_originalexp)))
    return x


def get_scaled_vector(a_vector):
    ''' scale a vector to be between 0 and 1
    a_vector: a numpy array
    '''
    ### scale the vector to be between 0 and 1
    a_vector_scaled = (a_vector - np.min(a_vector))/(np.max(a_vector) - np.min(a_vector))
    return a_vector_scaled