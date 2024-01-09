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
    


def get_metadata_ratLiver(data) -> tuple:
    """Return the metadata of the healthy rat liver dataset, including sample, strain and cluster information.
    data: AnnData object
    """

    #### sample metadata
    y_cluster = data.obs.cluster.squeeze()
    y_sample = data.obs[['sample']].squeeze()
    y_strain = data.obs.strain.squeeze()

    return y_sample, y_strain, y_cluster


def get_metadata_humanLiver(data) -> tuple:
    """Return the metadata of the healthy human liver dataset, including sample, cell type information.
    data: AnnData object
    """
    y_sample = data.obs['sample'].squeeze()
    y_cell_type = data.obs['cell_type'].squeeze()
    return y_sample, y_cell_type



def get_metadata_humanKidney(data) -> tuple:
    """Return the metadata of the healthy human kidney dataset, including sex, sampleID, cell type information.
    data: AnnData object
    """
    y_sample = data.obs['sampleID'].squeeze()
    y_cell_type = data.obs['Cell_Types_Broad'].squeeze()
    y_cell_type_sub = data.obs['Cell_Types_Subclusters'].squeeze()
    y_sex = data.obs['sex'].squeeze()

    return y_sample, y_sex, y_cell_type, y_cell_type_sub 



def get_metadata_humanPBMC(data) -> tuple:
    """Return the metadata of the stimulated human pbmc dataset, including sample, stimulation, cluster and cell type information.
    data: AnnData object
    """
    y_sample = data.obs['ind'].squeeze()
    y_stim = data.obs['stim'].squeeze()
    y_cell_type = data.obs['cell'].squeeze()
    y_cluster = data.obs['cluster'].squeeze()

    return y_sample, y_stim, y_cell_type, y_cluster 


def get_data_array(data) -> np.array:
    """Return the data matrix as a numpy array, and the number of cells and genes.
    data: AnnData object
    """

    data_numpy = data.X.toarray()

    ## working with the rat data
    num_cells = data_numpy.shape[0]
    num_genes = data_numpy.shape[1]

    genes = data.var_names

    print(num_cells, num_genes)

    return data_numpy, genes, num_cells, num_genes


def get_highly_variable_gene_indices(data_numpy, num_genes=const.num_genes, random=False):
    '''
    get the indices of the highly variable genes
    data_numpy: numpy array of the data (n_cells, n_genes)
    num_genes: number of genes to select
    random: whether to randomly select the genes or select the genes with highest variance
    '''
    if random:
        ### randomly select 1000 genes
        gene_idx = random.sample(range(0, data_numpy.shape[1]), num_genes)
    else:
        ### calculate the variance for each gene
        gene_vars = np.var(data_numpy, axis=0)
        ### select the top num_genes genes with the highest variance
        gene_idx = np.argsort(gene_vars)[-num_genes:]


    return gene_idx



def get_sub_data(data, num_genes=const.num_genes, random=False) -> tuple:    
    ''' subset the data matrix to the top num_genes genes
    y: numpy array of the gene expression matrix (n_cells, n_genes)
    random: whether to randomly select the genes or select the genes with highest variance
    num_genes: number of genes to select
    '''


    data_numpy = data.X.toarray()
    cell_sums = np.sum(data_numpy,axis=1) # row sums - library size
    gene_sums = np.sum(data_numpy,axis=0) # col sums - sum reads in a gene
    data = data[cell_sums!=0,gene_sums != 0] ## cells, genes

    data_numpy = data.X.toarray()
    ### calculate the variance for each gene
    gene_vars = np.var(data_numpy, axis=0)
    ### select the top num_genes genes with the highest variance
    gene_idx = np.argsort(gene_vars)[-num_genes:]

    #### select num_genes genes based on variance
    ## sort the gene_idx in ascending order
    gene_idx = np.sort(gene_idx)
    data = data[:,gene_idx]

    ### subset the data matrix to the top num_genes genes
    return data, gene_idx


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
        dict_covariate[column_level] = get_binary_covariate(data.obs[[a_metadata_col]].squeeze(), column_level)

    #### stack colummns of dict_covariate 
    x = np.column_stack((dict_covariate[column] for column in column_levels))
    return x



def get_lib_designmat(data, lib_size='nCount_RNA'): # nCount_originalexp for scMixology
    ''' return a design matrix for the library size covariate - equivalent to performing normalization
    data: AnnData object
    lib_size: the library size covariate name in the AnnData object
    '''
    x = np.column_stack((np.ones(data.shape[0]), np.array(data.obs[lib_size])))
    return x


def get_scaled_vector(a_vector):
    ''' scale a vector to be between 0 and 1
    a_vector: a numpy array
    '''
    ### scale the vector to be between 0 and 1
    a_vector_scaled = (a_vector - np.min(a_vector))/(np.max(a_vector) - np.min(a_vector))
    return a_vector_scaled


