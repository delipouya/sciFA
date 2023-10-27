#!/usr/bin/env Rscript
#############################################
#### implementing rotations using psych package
#############################################
library(psych)
library(mvtnorm)
library(tictoc)
# Set a random seed for reproducibility
set.seed(123)
factors = 15

num_genes <- 500
num_cells <- 1000


data <- LoadH5Seurat("~/sciFA/Data/scMix_5cl_merged.h5Seurat")
data <- SCTransform(data, assay='originalexp', variable.features.n =num_genes) 
var_genes = VariableFeatures(data, assay = 'SCT')
cells_to_keep = round(runif(num_cells, min=1, max=ncol(data)))
data_sub = data[var_genes, cells_to_keep] ### subset the samples and genes
dim(data_sub)

## input to psych: Number of observations * number of variables
data_exp = t(as.matrix(GetAssayData(data_sub, assay='SCT')))
data_exp[1:10,1:10]
dim(data_exp)

#############################################
# Factor analysis with rotations

tic()
fa_varimax <- fa(data_exp, nfactors = factors, rotate = "varimax")
time_varimax = toc()
print(paste0('varimax time: ', time_varimax))
saveRDS(fa_varimax, '~/sciFA/Results/psych_fa_varimax_sub.rds')


print('quartimax starting...')
tic()
fa_quartimax <- fa(data_exp, nfactors = factors, rotate = "quartimax")
time_quartimax = toc()
print(paste0('varimax time: ', time_varimax))
saveRDS(fa_quartimax, '~/sciFA/Results/psych_fa_quartimax_sub.rds')


print('equamax starting...')
tic()
fa_equamax <- fa(data_exp, nfactors = factors, rotate = "equamax")
time_equamax = toc()
print(paste0('equamax time: ', time_equamax))
saveRDS(fa_equamax, '~/sciFA/Results/psych_fa_equamax_sub.rds')


print('promax starting...')
tic()
fa_promax <- fa(data_exp, nfactors = factors, rotate = "promax")
time_promax = toc()
print(paste0('promax time: ', time_promax))
saveRDS(fa_promax, '~/sciFA/Results/psych_fa_promax_sub.rds')

print('oblimin starting...')
tic()
fa_oblimin <- fa(data_exp, nfactors = factors, rotate = "oblimin")
time_oblimin = toc()
print(paste0('oblimin time: ', time_oblimin))
saveRDS(fa_oblimin, '~/sciFA/Results/psych_fa_oblimin_sub.rds')



