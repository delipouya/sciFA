library(ggplot2)
library(ggpubr)
library(reshape2)
scale_minMax <- function(x){
  x_min = min(x)
  x_max = max(x)
  scaled = (x-x_min)/(x_max-x_min)
  return(scaled)
}

scale_Max <- function(x){
  x_max = max(x)
  scaled = (x)/(x_max)
  return(scaled)
}

importance_df_runtime = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded_RunTime.csv')
importance_df_runtime = importance_df_runtime[,-1]
colnames(importance_df_runtime) =  c("LogisticReg",  "DecisionTree", "RandomForest","XGB" ,"KNNpermute")
runtime_df = melt(importance_df_runtime)
ggplot(runtime_df, aes(y=variable, x=value))+geom_boxplot()+theme_classic()+
  theme(text = element_text(size=18))+xlab('')+xlab('time (min)')+ylab('')



metrics_df = read.csv('/home/delaram/sciFA/Results/scMix_3cl_merged_metrics.csv')
dim(metrics_df)
row.names(metrics_df) = paste0('F', 1:nrow(metrics_df))
head(metrics_df)
metrics_df = metrics_df[,colnames(metrics_df) != 'dip_pval']

metrics_df_scale = apply(metrics_df, MARGIN =2, scale) #1 indicates rows, 2 indicates columns,
row.names(metrics_df_scale) = row.names(metrics_df)

metrics_df_scale_minmax = apply(metrics_df, MARGIN =2, scale_minMax) #1 indicates rows, 2 indicates columns,
row.names(metrics_df_scale_minmax) = row.names(metrics_df)


metrics_df_scale_max = apply(metrics_df, MARGIN =2, scale_minMax) #1 indicates rows, 2 indicates columns,
row.names(metrics_df_scale_max) = row.names(metrics_df)


pheatmap::pheatmap(metrics_df, cluster_rows=F, cluster_cols = F, annotation_col = metric_annot,
                   border_color = 'black')
pheatmap::pheatmap(metrics_df_scale, cluster_rows=F, cluster_cols = F, annotation_col = metric_annot,
                   color = hcl.colors(50, "BluYl"),border_color = 'black')
pheatmap::pheatmap(metrics_df_scale_minmax, cluster_rows=F, cluster_cols = F, 
                   annotation_col = metric_annot, color = hcl.colors(50, "BluYl"),border_color = 'black')

pheatmap::pheatmap(metrics_df_scale_max, cluster_rows=F, cluster_cols = F, 
                   annotation_col = metric_annot, color = hcl.colors(50, "BluYl"),border_color = 'black')


metric_annot=data.frame(type=c(rep('specificity', 4), 'strength', rep('homogeneity', 4), rep('seperability', 16)))
rownames(metric_annot) = colnames(metrics_df_scale)

