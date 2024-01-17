data_file_path = '/home/delaram/sciFA//Data/inputdata_rat_set1_countData_2.h5ad'
library(SummarizedExperiment)

sce <- readH5AD(data_file_path)
class(assay(sce))

sce2 <- readH5AD(file, use_hdf5 = TRUE)



if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")
library(SeuratDisk)

Convert("example_dir/example_ad.h5ad", ".h5seurat")
# This creates a copy of this .h5ad object reformatted into .h5seurat inside the example_dir directory

# This .d5seurat object can then be read in manually

library(SeuratDisk)
SaveH5Seurat(dat, filename = "~/scLMM/input_data_designMat/inputdata_rat_set1_countData_2.h5seurat", overwrite = TRUE)
source_file = "~/scLMM/input_data_designMat/inputdata_rat_set1_countData_2.h5seurat"
dest_file = "~/scLMM/input_data_designMat/inputdata_rat_set1_countData_2.h5ad"
Convert(source_file, dest_file, assay="RNA", overwrite = TRUE)


seuratObject <- LoadH5Seurat(data_file_path)



merged_samples_cellb = readRDS('~/RatLiver/cell_browser/TLH_cellBrowser.rds')
head(merged_samples_cellb)
### adding the cell-type annotation ###
annotation_TLH = read.csv('~/RatLiver/figure_panel/TLH_annotation.csv')
meta_data = merged_samples_cellb@meta.data
meta_data$cell_id = rownames(meta_data)
meta_data2 = merge(meta_data, annotation_TLH, by.x='cluster', by.y='cluster', all.x=T, all.y=F)
meta_data2 = meta_data2[match(meta_data$cell_id, meta_data2$cell_id),]
sum(meta_data$cell_id != colnames(merged_samples_cellb))
sum(meta_data2$cell_id != colnames(merged_samples_cellb))
meta_data2$strain = gsub(meta_data2$strain, pattern = 'rat_', replacement = '')
rownames(meta_data2) = meta_data2$cell_id
meta_data2 <- meta_data2[,colnames(meta_data2) != 'cell_id']
head(meta_data2)
table(meta_data2$strain)

merged_samples_cellb@meta.data <- meta_data2
head(merged_samples_cellb@meta.data)
merged_samples_cellb@meta.data$umi = row.names(merged_samples_cellb@meta.data)
meta_data = merged_samples_cellb@meta.data
write.csv(meta_data, '/home/delaram/sciFA//Data/inputdata_rat_set1_metadat.csv')

df_umap <- data.frame(UMAP_1=getEmb(merged_samples, 'umap_h')[,1], 
                      UMAP_2=getEmb(merged_samples, 'umap_h')[,2])



########################### soupX decomtamination fraction
merged_data_soupX <- readRDS('~/RatLiver/Data/SoupX_data/TLH_decontaminated_merged_normed.rds')
head(merged_data_soupX)

sample_names = list.dirs('~/RatLiver/Data/SoupX_data/SoupX_inputs/',recursive = FALSE, full.names = FALSE)

for(i in 1:length(sample_names)){
  sample_name = sample_names[i]
  
  soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/default_param/', sample_name, '_soupX_out.rds'))
  #soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/paramPlus01/', sample_name, '_soupX_out_Rhoplus.0.1.rds'))
  #soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/paramPlus02/', sample_name, '_soupX_out_Rhoplus.0.2.rds'))
  #soup_profile = soupX_out$sc$soupProfile
}
head(soupX_out$sc$soupProfile)



