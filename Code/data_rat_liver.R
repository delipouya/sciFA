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




########################### soupX decomtamination fraction
merged_data_soupX <- readRDS('~/RatLiver/Data/SoupX_data/TLH_decontaminated_merged_normed.rds')
head(merged_data_soupX)

sample_names = list.dirs('~/RatLiver/Data/SoupX_data/SoupX_inputs/',recursive = FALSE, full.names = FALSE)

soup_profile_list = list(NA, NA, NA, NA)
for(i in 1:length(sample_names)){
  sample_name = sample_names[i]
  
  soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/default_param/', sample_name, '_soupX_out.rds'))
  #soupX_out = readRDS(paste0('~/RatLiver/Data/SoupX_data/SoupX_outputs/paramPlus01/', sample_name, '_soupX_out_Rhoplus.0.1.rds'))
  
  soup_profile = data.frame(genes=row.names(soupX_out$sc$soupProfile),soup=soupX_out$sc$soupProfile$est)
  soup_profile = soup_profile[order(soup_profile$soup, decreasing = T),]
  soup_profile_list[[i]] = soup_profile 
}
names(soup_profile_list) = sample_names 
num_genes = 50
consisitent_soup_genes_df = data.frame(table(unlist(lapply(soup_profile_list, function(x) x$genes[1:num_genes]))))
top_soup_genes = as.character(consisitent_soup_genes_df$Var1[consisitent_soup_genes_df$Freq>1])


############# comparing sciRED based strain differences with DE based comparison
pca_scores_varimax_df_merged = read.csv('~/sciFA//Results/pca_scores_varimax_df_ratliver_libReg.csv')
varimax_loading_df = read.csv('~/sciFA/Results/varimax_loading_df_ratliver_libReg.csv')

head(pca_scores_varimax_df_merged)
head(varimax_loading_df)

varimax_loading_df_ord = varimax_loading_df[order(varimax_loading_df$F23, decreasing = F),]
varimax_loading_df_ord = data.frame(genes=varimax_loading_df_ord$X,factor=varimax_loading_df_ord$F23)
head(varimax_loading_df_ord,40)
head(varimax_loading_df_ord,20)
varimax_loading_df_ord[varimax_loading_df_ord$genes=='Itgal',]



table(merged_samples_cellb$annotation) 
clusters_to_include = c(  'Marco/Cd5l Mac (5)') #'Marco/Cd5l Mac (10)', 
merged_samples_sub = merged_samples_cellb[,merged_samples_cellb$annotation %in% clusters_to_include]
dim(merged_samples_sub)

install.packages('devtools')
devtools::install_github('immunogenomics/presto')

table(merged_samples_sub$strain)
Idents(merged_samples_sub) = merged_samples_sub$strain
strain_markers = FindMarkers(merged_samples_sub, ident.1 = 'DA', ident.2 = 'LEW')
head(strain_markers,20)
strain_markers$score = -(strain_markers$avg_log2FC * log(strain_markers$p_val_adj))
strain_markers_sort = strain_markers[order(strain_markers$score, decreasing = F),]
head(strain_markers_sort,20)
write.csv(strain_markers, '~/sciFA/Results/strain_markers_ratliver_DE_strain_nonInf_cluster5.csv')



varimax_genes_strain = c(head(varimax_loading_df_ord$genes,20), tail(varimax_loading_df_ord$genes,20))
row.names(strain_markers)[1:25][row.names(strain_markers)[1:25] %in% top_soup_genes]
varimax_genes_strain[varimax_genes_strain%in% top_soup_genes]
#its hard to claim b2m and pck1.
