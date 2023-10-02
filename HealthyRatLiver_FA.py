import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import abline_plot
import functions_plotting as fplot
import functions_processing as fproc

#### calculate the goodness of fit of GLM model
# https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html#statsmodels.genmod.generalized_linear_model.GLMResults

### statsmodels GLM source code:
# https://github.com/statsmodels/statsmodels/blob/main/statsmodels/genmod/families/links.py

### poisson family link function
# https://github.com/statsmodels/statsmodels/blob/main/statsmodels/genmod/families/family.py


data_file_path = '/home/delaram/scLMM/input_data_designMat/inputdata_rat_set1_countData_2.h5ad'
data = fproc.import_AnnData(data_file_path)
y, num_cells, num_genes = fproc.get_data_array(data)
y_sample, y_strain, y_cluster = fproc.get_metadata_ratLiver(data)
y = fproc.get_sub_data(y, random=False) # subset the data to num_genes HVGs

#### design matrix - library size only
x = fproc.get_lib_designmat(data, lib_size='nCount_RNA')

#### design matrix - library size and sample
x_sample = fproc.get_design_mat('sample', data) ## TODO: check why this is not working
x = np.column_stack((data.obs.nCount_RNA, x_sample)) #, x_batch
x = sm.add_constant(x) ## adding the intercept





