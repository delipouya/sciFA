library(reshape2)
library(ggplot2)
### import the results from the simulation (simulation_fc_multi.py) and visualize the results
df = read.csv('~/sciFA/metric_overlap_corr_df_sim20.csv')
df = read.csv('~/sciFA/metric_overlap_corr_df_sim20_v2.csv')
df = read.csv('~/sciFA/Code/metric_overlap_corr_df_sim100_v3_nov1.csv')

df = data.frame(t(df))
colnames(df) = df[1,]
df = df[-1,]
head(df)

cols_to_remove = c('factor_gini_meanImp', 'factor_gini_AUC', 'factor_entropy_meanImp', 'factor_entropy_AUC', 'factor_simpon_meanImp', 'factor_simpson_AUC')
df = df[,!colnames(df) %in% cols_to_remove]
head(df)

df_melt = melt(t(df))
colnames(df_melt) = c('metric', 'overlap', 'R')
df_melt$R = as.numeric(df_melt$R)

ggplot(df_melt, aes(x=metric,y=R))+geom_boxplot(notch = TRUE, fill='maroon')+
  coord_flip()+ylab('Correlation with overlap value')+theme_classic()

df.t = data.frame(t(data.frame(df)))
df.t_num = data.frame(lapply(df.t, as.numeric))
row.names(df.t_num) = rownames(df.t)

df.t_num$mean = rowMeans(df.t_num)
df.t_num$sd = apply(df.t_num, 1, sd, na.rm=TRUE)

head(df.t_num)
ncol(df.t_num)

dev.off()
gridExtra::grid.table(round(df.t_num[,c(21,22)],3))
View(round(df.t_num[,c(21,22)],3))
