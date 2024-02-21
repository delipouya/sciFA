load('~/HumanLiver/extra_files/inst/liver/HumanLiver.RData')
load('~/HumanLiver/extra_files/inst/liver/HumanLiver_savedRes.RData')
HumanLiverSeurat = UpdateSeuratObject(HumanLiverSeurat)

##### calculating the mt percentage
#HumanLiverSeurat[["percent.mt"]] <- PercentageFeatureSet(HumanLiverSeurat, pattern = "^MT-")
mt_indices = grep('^MT-',rownames(HumanLiverSeurat))
sum_MT_conuts = colSums(GetAssayData(HumanLiverSeurat, layer = 'counts')[mt_indices,])
sum_counts = colSums(GetAssayData(HumanLiverSeurat, layer = 'counts'))
HumanLiverSeurat[["percent.mt"]]  = sum_MT_conuts/sum_counts

################################################################
################ Saving the object to be used in Python
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
################################################################################



tsne_df = Embeddings(HumanLiverSeurat, 'tsne')
tsne_df = cbind(tsne_df, HumanLiverSeurat@meta.data)
##### reading factor
factor_loading = read.csv('/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv')
factor_scores = read.csv('/home/delaram/sciFA/Results/factor_scores_umap_df_humanlivermap.csv')
factor_scores = factor_scores[factor_scores$id %in% row.names(tsne_df),]

head(factor_scores)
dim(factor_scores)

sum(factor_scores$id != row.names(tsne_df))
tsne_df_merged = cbind(tsne_df, factor_scores)
head(tsne_df_merged)

library(RColorBrewer)
n <- 30
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

library(ggplot2)
colnames(tsne_df_merged)
tsne_df_merged_2 = tsne_df_merged[,-c(45:51)]
#tsne_df_merged_2 = tsne_df_merged
ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=cell_type))+geom_point(size=2)+theme_classic()+scale_color_manual(values = col_vector)
ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=cell_type))+geom_point(size=0.6)+theme_classic()+scale_color_manual(values = col_vector)

ggplot(tsne_df_merged_2, aes(factor_2,factor_3,color=cell_type))+
  geom_point(size=0.6)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F3')+ylab('F4')

ggplot(tsne_df_merged_2, aes(factor_8,factor_12,color=cell_type))+
  geom_point(size=0.6)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F9')+ylab('F13')

ggplot(tsne_df_merged_2, aes(factor_20,factor_23,color=cell_type))+
  geom_point(size=0.8)+theme_classic()+scale_color_manual(values = col_vector)+xlab('F21')+ylab('F24')

ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=factor_9))+geom_point(size=0.6,alpha=0.6)+
  theme_classic()+scale_color_viridis_b(option='plasma',direction = -1)
ggplot(tsne_df_merged_2, aes(tSNE_1,tSNE_2,color=factor_18))+geom_point(alpha=0.7)+
  theme_classic()+scale_color_viridis_b(direction = -1)
ggplot(tsne_df_merged_2, aes(umap_1,umap_2,color=factor_29))+geom_point(alpha=0.7)+
  theme_classic()+scale_color_viridis_b(direction = -1)

qc_columns = c('total_counts', 'total_features' , 'percent.mt','S.Score', 'G2M.Score')
factor_cols = paste0('factor_', 0:29)
factor_cols = paste0('factor_',c(0, 9, 18, 19, 21, 25, 27, 28, 29))

c(qc_columns, factor_cols) %in% colnames(tsne_df_merged_2)
tsne_df_merged_3 = tsne_df_merged_2[, colnames(tsne_df_merged_2) %in% c(qc_columns, factor_cols)]
head(tsne_df_merged_3)
colnames(cor(tsne_df_merged_3))
cor_mat = cor(tsne_df_merged_3)[qc_columns, factor_cols]
library(pheatmap)
# make the color pallete
clrsp <- colorRampPalette(c("darkgreen", "white", "purple"))   
clrs <- clrsp(200) 
breaks1 <- seq(-1, 1, length.out = 200)
colnames(cor_mat) = paste0('F',c(0, 9, 18, 19, 21, 25, 27, 28, 29)+1)
pheatmap(cor_mat, cluster_cols = F, breaks = breaks1, color =  clrs, display_numbers = T)


###### evaluating loadings
source('~/RatLiver/Codes/Functions.R')
Initialize()
library(gprofiler2)

get_gprofiler_enrich <- function(markers, model_animal_name){
  gostres <- gost(query = markers,
                  ordered_query = TRUE, exclude_iea =TRUE, 
                  sources=c('GO:BP' ,'REAC'),
                  organism = model_animal_name)
  return(gostres)
}


genes = read.csv('/home/delaram/sciFA/Results/genes_humanlivermap.csv')
df = data.frame(gene= genes$X0,factor=factor_loading$X29)
model_animal_name ='hsapiens'

df_pos = df[order(df$factor, decreasing = T),]
head(df_pos,10)
num_genes = 200

enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], model_animal_name)
head(enrich_res$result,10)
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos = enrich_res_pos[1:20,]
enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'p_value')]
enrich_res_pos$log_p = -log(enrich_res_pos$p_value)
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(paste0('(positive loading)'))


df_neg = df[order(df$factor, decreasing = F),]
head(df_neg,10)
enrich_res = get_gprofiler_enrich(markers=df_neg$gene[1:num_genes], model_animal_name)
head(enrich_res$result,30)
enrich_res_neg = data.frame(enrich_res$result)
enrich_res_neg = enrich_res_neg[1:20,]
enrich_res_neg = enrich_res_neg[,colnames(enrich_res_neg) %in% c('term_name', 'p_value')]
enrich_res_neg$log_p = -log(enrich_res_neg$p_value)
ggplot(enrich_res_neg, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(paste0('(negative loading)'))


ggplot(factor_scores, aes(x=cell_type, y=factor_18, fill=cell_type))+geom_boxplot()+
  coord_flip()+scale_fill_manual(values = col_vector)+theme_classic()+xlab('')+ylab('F19 Score')

