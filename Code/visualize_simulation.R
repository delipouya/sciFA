library(reshape2)
library(ggplot2)
### import the results from the simulation (simulation_fc_multi.py) and visualize the results
df = read.csv('~/sciFA/metric_overlap_corr_df_sim20.csv')
df = read.csv('~/sciFA/metric_overlap_corr_df_sim20_v2.csv')
df = read.csv('~/sciFA/Code/metric_overlap_corr_df_sim100_v3_nov1.csv')
df = read.csv('~/sciFA/Code/metric_overlap_corr_df_sim100_Jan2024.csv')
df = read.csv('~/sciFA/Results//metric_overlap_corr_df_sim100_Jan2024_v2.csv')


df = data.frame(t(df))
colnames(df) = df[1,]
df = df[-1,]
head(df)
colnames(df)
cols_to_remove = c('factor_gini_meanImp', 'factor_gini_AUC', 
                   'factor_entropy_meanImp', 'factor_entropy_AUC',
                   'factor_simpon_meanImp', 'factor_simpson_AUC', 'dip_pval'
                   #"1-AUC_arith","1-AUC_geo"
                   #'ASV_simpson','ASV_entropy'
                   )
df = df[,!colnames(df) %in% cols_to_remove]
head(df)

df_melt = melt(t(df))
colnames(df_melt) = c('metric', 'overlap', 'R')
df_melt$R = as.numeric(df_melt$R)
head(df_melt)
bimodality_metric=c( "bic_km","calinski_harabasz_km", "davies_bouldin_km","silhouette_km",
   "vrs_km","wvrs_km","bic_gmm", "silhouette_gmm",
   "vrs_gmm","wvrs_gmm","likelihood_ratio","bimodality_index","dip_score",
   "kurtosis","outlier_sum" )
heterogeneity_metric=c("ASV_arith","ASV_geo","1-AUC_arith","1-AUC_geo",'ASV_simpson','ASV_entropy')
effect_size_metric=c('factor_variance')
df_melt$metric_type[df_melt$metric %in% bimodality_metric]='separability'
df_melt$metric_type[df_melt$metric %in% heterogeneity_metric]='heterogeneity'
df_melt$metric_type[df_melt$metric %in% effect_size_metric]='effect size'


ggplot(df_melt, aes(x=metric,y=R))+geom_boxplot(notch = TRUE, fill='maroon')+
  coord_flip()+ylab('Correlation with overlap value')+
  theme(text = element_text(size=16),
        axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 15, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 15, angle = 90, hjust = .5, vjust = .5, face = "plain"))


ggplot(df_melt, aes(x=metric,y=R,fill=metric_type))+
  geom_boxplot(notch = TRUE)+xlab('')+
  coord_flip()+ylab('Correlation with overlap value')+
  theme(text = element_text(size=16),
      axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
      axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
      axis.title.x = element_text(color = "grey20", size = 14, angle = 0, hjust = .5, vjust = 0, face = "plain"),
      axis.title.y = element_text(color = "grey20", size = 14, angle = 90, hjust = .5, vjust = .5, face = "plain"))


df.t = data.frame(t(data.frame(df)))
df.t_num = data.frame(lapply(df.t, as.numeric))
row.names(df.t_num) = rownames(df.t)

df.t_num$mean = rowMeans(df.t_num)
df.t_num$sd = apply(df.t_num, 1, sd, na.rm=TRUE)

head(df.t_num)
ncol(df.t_num)

dev.off()
gridExtra::grid.table(round(df.t_num[,c('mean','sd')],3))
gridExtra::grid.table(round(df.t_num,3))

View(round(df.t_num[,c('mean','sd')],3))
