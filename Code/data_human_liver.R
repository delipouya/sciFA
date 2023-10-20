load('~/HumanLiver/inst/liver/HumanLiver.RData')
load('~/HumanLiver/inst/liver/HumanLiver_savedRes.RData')
seurat_obj <- HumanLiverSeurat

seur <- CreateSeuratObject(GetAssayData(HumanLiverSeurat, 'counts'))
seur@meta.data = HumanLiverSeurat@meta.data

# The following 13086 features requested have not been scaled
annotations=c('Hep1','abT cell','Hep2','infMac','Hep3','Hep4','plasma cell',
              'NK-like cell','gdT cell1','nonInfMac','periportal LSEC',
              'central venous LSEC','portal Endothelial cell','Hep5','Hep6',
              'mature B cell','cholangiocyte','gdT cell2','erythroid cell',
              'hepatic stellate cell')

label_df = data.frame(cluster=paste0('cluster_',1:20),labels=annotations)
Idents(seur) = paste0('cluster_', as.character(seur$res.0.8))
human_liver_annot = data.frame(umi=colnames(seur), cluster=Idents(seur))
human_liver_annot = merge(human_liver_annot, label_df, by.x='cluster', by.y='cluster', all.x=T, sort=F)

human_liver_annot_sorted <- human_liver_annot[match(colnames(seur), human_liver_annot$umi),]
sum(human_liver_annot_sorted$umi != colnames(seur))
seur$cell_type = human_liver_annot_sorted$labels

seur$sample = unlist(lapply(strsplit(colnames(seur), '_'), '[[', 1))
head(seur)
dim(seur)

SaveH5Seurat(seur, filename ='~/sciFA/Data/HumanLiverAtlas.h5Seurat' ,overwrite = TRUE)
Convert('~/sciFA/Data/HumanLiverAtlas.h5Seurat', dest = "h5ad")
