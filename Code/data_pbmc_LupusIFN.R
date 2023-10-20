######## preparing the data within the Muscat dataset 
# 10x droplet-based scRNA-seq PBMC data from 8 Lupus patients before and after 6h-treatment with INF-beta 
# https://github.com/HelenaLC/muscData
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583

#BiocManager::install("muscData")
library(muscData)
library(Seurat)
library(SeuratData)
library(SeuratDisk)

Kang18_8vs8 = muscData::Kang18_8vs8(metadata = FALSE)
Kang18_8vs8$multiplets
Kang18_8vs8@assays[['counts']]
class(Kang18_8vs8)
Kang18_8vs8_seur = as.Seurat(Kang18_8vs8, counts = "counts", data = "counts")
GetAssayData(Kang18_8vs8_seur)
dim(Kang18_8vs8_seur)
SaveH5Seurat(Kang18_8vs8_seur, filename = "~/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5Seurat")
Convert("~/sciFA/Data/PBMC_Lupus_Kang8vs8_data.h5Seurat", dest = "h5ad")






