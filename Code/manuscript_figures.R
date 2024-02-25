library(ggplot2)

################ Kidney figure scatter plots
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_merged_kidneyMap.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_kidneyMap.csv')

head(pca_scores_varimax_df_merged)
ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F18, color=sex))+geom_point(size=4)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-18')+scale_color_manual(values=c('palevioletred', 'skyblue'))

library(RColorBrewer)
n <- 60
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))

ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F18, color=sampleID))+geom_point(size=0.3)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),legend.position="none",
        legend.box.background = element_rect(colour = "black"), legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-18')+scale_color_manual(values=col_vector)


################ PBMC stimulated dataset figure scatter plots
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_merged_lupusPBMC.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_lupusPBMC.csv')

head(pca_scores_varimax_df_merged)
ggplot(pca_scores_varimax_df_merged[pca_scores_varimax_df_merged$F9>(-25),], aes(x = F1, y = F9, color=cell))+geom_point(size=0.7)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  #scale_color_brewer(palette = "Set2")+
  scale_color_manual(values = col_vector)+
  xlab('Factor-1')+ylab('Factor-9')

ggplot(pca_scores_varimax_df_merged[pca_scores_varimax_df_merged$F2>(-35),], aes(x = F1, y = F2, color=stim))+geom_point(size=0.6)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-2')+ scale_color_brewer(palette = "Set2")



ggplot(pca_scores_varimax_df_merged[pca_scores_varimax_df_merged$F2>(-35),], aes(x = cell, y = F2, color=cell))+geom_boxplot()+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  ylab('Factor-2')+scale_color_manual(values=col_vector)+coord_flip()


ggplot(pca_scores_varimax_df_merged[pca_scores_varimax_df_merged$F9>(-25),], aes(x = cell, y = F2, color=cell))+geom_boxplot()+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  ylab('Factor-9')+scale_color_manual(values=col_vector)+coord_flip()



############### rat liver figure scatter plots
library(Seurat)
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_merged_ratLiver.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_ratLiver.csv')

merged_samples_cellb = readRDS('~/RatLiver/cell_browser/TLH_cellBrowser.rds')
umap_df = Embeddings(merged_samples_cellb, 'umap')
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
merged_samples_cellb@meta.data <- meta_data2
head(merged_samples_cellb@meta.data)

#### adding UMAP embedding to the merged sample dataframe
merged_samples_cellb$UMAP_1 = umap_df[,1]
merged_samples_cellb$UMAP_2 = umap_df[,2]

head(merged_samples_cellb@meta.data)
head(pca_scores_varimax_df_merged)

table(merged_samples_cellb@meta.data$sample_name)
table(pca_scores_varimax_df_merged$sample)
merged_samples_cellb$sample_2 = merged_samples_cellb@meta.data$sample_name
merged_samples_cellb$ID = colnames(merged_samples_cellb)
merged_samples_cellb$ID = gsub('rat_DA_01_reseq', 'DA_01',merged_samples_cellb$ID)
merged_samples_cellb$ID = gsub('rat_DA_M_10WK_003', 'DA_02',merged_samples_cellb$ID)
merged_samples_cellb$ID = gsub('rat_Lew_01', 'LEW_01',merged_samples_cellb$ID)
merged_samples_cellb$ID = gsub('rat_Lew_02', 'LEW_02',merged_samples_cellb$ID)
sum(!merged_samples_cellb$ID %in% pca_scores_varimax_df_merged$refined_cell_ID)

merged_df = merge(pca_scores_varimax_df_merged, merged_samples_cellb@meta.data, by.x='refined_cell_ID', by.y='ID')
dim(merged_df)
dim(pca_scores_varimax_df_merged)

head(pca_scores_varimax_df_merged)
head(merged_df)
ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F6, color=strain))+geom_point(size=1)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-6')+scale_color_manual(values=c('goldenrod', 'grey50'))

ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F20, color=strain))+geom_point(size=1)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-20')+scale_color_manual(values=c('goldenrod', 'grey50'))


ggplot(merged_df, aes(UMAP_1,UMAP_2,color=F19))+geom_point(alpha=0.7,size=1)+
  theme_classic()+#scale_color_viridis_b(name = "Factor-20\nScore",direction = +1, values = rescale(c(rep(0, 50),51:100)))+
  theme(axis.text.x = element_text(size = 16), axis.text.y = element_text(size = 16),
        axis.title.y = element_text(size = 18), axis.title.x = element_text(size = 18))+
  scale_color_stepsn(colours =c(rep('#440154FF',1),
                                '#404688FF', '#3B528BFF','#365D8DFF','#31688EFF', '#2C728EFF',
                                '#287C8EFF','#24868EFF','#21908CFF','#1F9A8AFF', '#20A486FF',
                                rep('#21908CFF',1), rep('#ffcf20FF',1)))

merged_df[merged_df$F19>15,]

library(RColorBrewer)
n <- length(names(table(pca_scores_varimax_df_merged$annotation)))
qual_col_pals = brewer.pal.info[brewer.pal.info$category == 'qual',]
col_vector = unlist(mapply(brewer.pal, qual_col_pals$maxcolors, rownames(qual_col_pals)))


ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F6, color=annotation))+geom_point(size=1)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-6')+scale_color_manual(values=col_vector)



ggplot(pca_scores_varimax_df_merged, aes(x = F1, y = F20, color=annotation))+geom_point(size=3)+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  xlab('Factor-1')+ylab('Factor-20')+scale_color_manual(values=col_vector)


ggplot(pca_scores_varimax_df_merged, aes(x = annotation, y = F20, color=annotation))+geom_boxplot()+theme_classic()+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"), 
        legend.text=element_text(size=12),#legend.title =element_text(size=16),
        legend.background = element_blank(),
        #legend.box.background = element_rect(colour = "black"), 
        legend.title=element_blank())+
  +ylab('Factor-20')+scale_color_manual(values=col_vector)+coord_flip()




############### rat liver figure scatter plots
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_merged_ratLiver.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_ratLiver.csv')

'/home/delaram/sciFA/Results/factor_scores_umap_df_humanlivermap.csv'
'/home/delaram/sciFA/Results/factor_loading_humanlivermap.csv'