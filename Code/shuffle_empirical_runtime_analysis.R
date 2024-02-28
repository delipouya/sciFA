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

importance_df_runtime = read.csv('/home/delaram/sciFA/Results/results2023/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded_RunTime.csv')
test = read.csv('/home/delaram/sciFA/Results/shuffle_empirical_dist_time/time_df_scMixology_varimax_shuffle_0.csv')

test = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/time_v1/time_df_human_liver_pearson_shuffle_1.csv')
head(test)
test = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/time/time_df_scMixology_pearson_shuffle_0.csv')
head(test)

head(importance_df_runtime)
importance_df_runtime = importance_df_runtime[,-1]
colnames(importance_df_runtime) =  c("LogisticReg",  "DecisionTree", "RandomForest","XGB" ,"KNNpermute")
runtime_df = melt(importance_df_runtime)
ggplot(runtime_df, aes(y=variable, x=value))+geom_boxplot()+theme_classic()+
  theme(text = element_text(size=18))+xlab('')+xlab('Run time (min)')+ylab('')



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





####################################################################
############## Updated runtime analysis for shuffle analysis
########################################################

test = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/time_v1/time_df_human_liver_pearson_shuffle_1.csv')
test = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/time/time_df_scMixology_pearson_shuffle_0.csv')
head(test)

file = '/home/delaram/sciFA/Results/benchmark/pearson/time/'
file = '/home/delaram/sciFA/Results/benchmark_humanliver/pearson/time//'
time_list = lapply(list.files(file, pattern = "time_df*", full.names = T)[1:500], read.csv)
time_list_df <- Reduce(rbind,time_list)
head(time_list_df)
time_list_df = time_list_df[,-1]
head(time_list_df)
time_list_df_m = melt(time_list_df)
head(time_list_df_m)

ggplot(time_list_df_m, aes(y=value, x=variable))+geom_boxplot()+
  theme_classic()+coord_flip()+
  #+scale_fill_brewer(palette = "Set2")+
  theme(text = element_text(size=17),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1,size = 19),
        axis.title.x = element_text(angle = 0,size = 23),
        axis.text.y = element_text(vjust = 0.5, hjust=1,size = 19))+
  ylab('Run Time (min)')+xlab('')+ggtitle('')




