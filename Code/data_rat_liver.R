merged_samples_cellb = readRDS('~/RatLiver/cell_browser/TLH_cellBrowser.rds')
head(merged_samples_cellb)
library(Seurat)
umap_df = Embeddings(merged_samples_cellb, 'umap')
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

cbind(umap_df)

################################################################################# 
########################### soupX decomtamination fraction
###########################  Saving output - DO NOT RUN !!! ########################### 
merged_data_soupX <- readRDS('~/RatLiver/Data/SoupX_data/TLH_decontaminated_merged_normed.rds')
head(merged_data_soupX)

sample_names = list.dirs('~/RatLiver/Data/SoupX_data/SoupX_inputs/',recursive = FALSE, full.names = FALSE)

soup_profile_list = list(NA, NA, NA, NA)
for(i in 1:length(sample_names)){
  sample_name = sample_names[i]
  
  soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/default_param/', 
                             sample_name, '_soupX_out.rds'))
  #soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/paramPlus01/', sample_name, '_soupX_out_Rhoplus.0.1.rds'))
  
  soup_profile = data.frame(genes=row.names(soupX_out$sc$soupProfile),soup=soupX_out$sc$soupProfile$est)
  soup_profile = soup_profile[order(soup_profile$soup, decreasing = T),]
  soup_profile_list[[i]] = soup_profile 
}
names(soup_profile_list) = sample_names 
rm(soupX_out)
gc()

for(i in 1:length(soup_profile_list)){
  soup_df_i = soup_profile_list[[i]]
  print(head(soup_df_i))
  sample_name = sample_names[i]
  write.csv(soup_df_i,
            paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/default_param/', 
                   sample_name, '_soupX_profile_df.csv'))
}

###################################################### 
###########################  Reading the output ########################### 
##################################################### 

sample_names = list.dirs('~/RatLiver/Data/SoupX_data/SoupX_inputs/',
                         recursive = FALSE, full.names = FALSE)
soup_profile_list = list(NA, NA, NA, NA)

for(i in 1:length(sample_names)){
  sample_name = sample_names[i]
  soup_df_i = read.csv(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/default_param/', 
                   sample_name, '_soupX_profile_df.csv'))
  soup_profile_list[[i]] = soup_df_i
}

num_genes = 50
consisitent_soup_genes_df = data.frame(table(unlist(lapply(soup_profile_list, function(x) x$genes[1:num_genes]))))
top_soup_genes = as.character(consisitent_soup_genes_df$Var1[consisitent_soup_genes_df$Freq>2])


############# comparing sciRED based strain differences with DE based comparison
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_ratliver_libReg_10000.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_ratliver_libReg_10000.csv')

pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_merged_ratLiver.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_ratLiver.csv')

head(pca_scores_varimax_df_merged)
head(varimax_loading_df)

varimax_loading_df_ord = varimax_loading_df[order(varimax_loading_df$F20, decreasing = T),]
varimax_loading_df_ord = data.frame(genes=varimax_loading_df_ord$X,factor=varimax_loading_df_ord$F20)
head(varimax_loading_df_ord,20)
library(tidyverse)
tail(varimax_loading_df_ord,20) %>% map_df(rev)
varimax_loading_df_ord[varimax_loading_df_ord$genes=='Itgal',]


varimax_genes_strain = c(head(varimax_loading_df_ord$genes,20), tail(varimax_loading_df_ord$genes,20))
varimax_genes_strain[varimax_genes_strain%in% top_soup_genes]

###### create table for manuscript figures
table_to_vis = data.frame(head(varimax_loading_df_ord,20))
table_to_vis = data.frame(tail(varimax_loading_df_ord,20) %>% map_df(rev))
rownames(table_to_vis) = NULL
colnames(table_to_vis) = c('Gene', 'Score')
table_to_vis$Score = round(table_to_vis$Score, 3)
#table_to_vis[2,]$Gene = 'APOBEC3A'
library(gridExtra)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(table_to_vis, theme=tt2)


#varimax_ratliver_F20_pos

##################################
table(merged_samples_cellb$annotation) 
clusters_to_include = c(  'Marco/Cd5l Mac (5)') #'Marco/Cd5l Mac (10)', 
merged_samples_sub = merged_samples_cellb[,merged_samples_cellb$annotation %in% clusters_to_include]
dim(merged_samples_sub)

#install.packages('devtools')
#devtools::install_github('immunogenomics/presto')

table(merged_samples_sub$strain)
Idents(merged_samples_sub) = merged_samples_sub$strain
strain_markers = FindMarkers(merged_samples_sub, ident.1 = 'DA', ident.2 = 'LEW')
head(strain_markers,20)
strain_markers$score = -(strain_markers$avg_log2FC * log(strain_markers$p_val_adj))
strain_markers_sort = strain_markers[order(strain_markers$score, decreasing = F),]
head(strain_markers_sort,20)
strain_markers <- read.csv('~/sciFA/Results/strain_markers_ratliver_DE_strain_nonInf_cluster5.csv')

strain_markers$X[1:25][strain_markers$X[1:25] %in% top_soup_genes]
#its hard to claim b2m and pck1.
strain_markers[1:25,]
strain_markers_sub = strain_markers[,!colnames(strain_markers) %in% c('p_val', 'pct.1' ,'pct.2')]
head(strain_markers_sub)
colnames(strain_markers_sub)[1] = c('Gene')
strain_markers_sub$avg_log2FC = round(strain_markers_sub$avg_log2FC, 3)
head(strain_markers_sub)
strain_markers_vis = strain_markers_sub[1:25,]

split_list = str_split(strain_markers_vis$p_val_adj, 'e')
first = round(as.numeric(unlist(lapply(split_list, '[[', 1))),2)
second = unlist(lapply(split_list, '[[', 2))
strain_markers_vis$p_val_adj = paste0(first, 'e', second)

#table_to_vis[2,]$Gene = 'APOBEC3A'
library(gridExtra)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(strain_markers_vis, theme=tt2)



head(strain_markers)
strain_markers$ambient = ifelse(strain_markers$X %in% top_soup_genes, 'ambientRNA','')
head(strain_markers)
strain_markers$log_pval_adj = -log10(strain_markers$p_val_adj)
ggplot(strain_markers, aes(avg_log2FC, log_pval_adj, color=ambient))+geom_point(size=1.3,alpha=0.7)+theme_classic()+
  scale_color_manual(values = c('grey30','red'))+
  geom_text(data=subset(strain_markers, log_pval_adj> 40 & ambient=='ambientRNA' &(log_pval_adj > 2 | log_pval_adj <(-2))),
            aes(avg_log2FC, log_pval_adj,label=X),check_overlap = T,cex=4,col='black',fontface='bold',vjust = -0.3, nudge_y = 1.7)+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Average log2FC')+ylab('-log10(adj pvalue)')
