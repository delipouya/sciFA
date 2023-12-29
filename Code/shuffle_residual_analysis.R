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
###############################################################################################
########################## importance evaluation for model comparison
################################################################################################
importance_df_m_merged_deviance = read.csv('/home/delaram/sciFA/Results/residual/importance_df_melted_scMixology_varimax_baseline_deviance.csv')
importance_df_m_merged_deviance$res = 'deviance'
importance_df_m_merged_pearson = read.csv('/home/delaram/sciFA/Results/residual/importance_df_melted_scMixology_varimax_baseline_pearson.csv')
importance_df_m_merged_pearson$res = 'pearson'
importance_df_m_merged_response = read.csv('/home/delaram/sciFA/Results/residual/importance_df_melted_scMixology_varimax_baseline_response.csv')
importance_df_m_merged_response$res = 'response'
dim(importance_df_m_merged_response)

importance_df_residual_baseline = rbind(rbind(importance_df_m_merged_response, importance_df_m_merged_pearson), importance_df_m_merged_deviance)
head(importance_df_residual_baseline)
importance_df_residual_baseline$type = 'baseline'
dim(importance_df_residual_baseline)

file = '/home/delaram/sciFA/Results/residual/pearson/'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T), read.csv)
imp_shuffle_pearson <- Reduce(rbind,imp_list)
imp_shuffle_pearson$res = 'pearson'
head(imp_shuffle_pearson)

file = '/home/delaram/sciFA/Results/residual/response/'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T), read.csv)
imp_shuffle_response <- Reduce(rbind,imp_list)
imp_shuffle_response$res = 'response'
head(imp_shuffle_response)

file = '/home/delaram/sciFA/Results/residual/deviance//'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T), read.csv)
imp_shuffle_deviance <- Reduce(rbind,imp_list)
imp_shuffle_deviance$res = 'deviance'
head(imp_shuffle_deviance)
dim(imp_shuffle_deviance)

importance_df_residual_shuffle = rbind(rbind(imp_shuffle_pearson, imp_shuffle_response), imp_shuffle_deviance)
head(importance_df_residual_shuffle)
importance_df_residual_shuffle$type = 'shuffle'

library(ggplot2)
importance_df_residual_merged = rbind(importance_df_residual_baseline, importance_df_residual_shuffle)
importance_df_residual_merged$importance_abs = abs(importance_df_residual_merged$importance)
ggplot2::ggplot(importance_df_residual_merged, aes(y=importance_abs, x=res, fill=type))+geom_boxplot()+theme_classic()+coord_flip()



###############################################################################################
########################## importance evaluation for model comparison
################################################################################################
meanimp_df_merged_deviance = read.csv('/home/delaram/sciFA/Results/residual/meanimp_df_scMixology_varimax_baseline_deviance.csv')
meanimp_df_merged_deviance$res = 'deviance'
meanimp_df_merged_pearson = read.csv('/home/delaram/sciFA/Results/residual/meanimp_df_scMixology_varimax_baseline_pearson.csv')
meanimp_df_merged_pearson$res = 'pearson'
meanimp_df_merged_response = read.csv('/home/delaram/sciFA/Results/residual/meanimp_df_scMixology_varimax_baseline_response.csv')
meanimp_df_merged_response$res = 'response'

meanimp_df_residual_baseline = rbind(rbind(meanimp_df_merged_response, meanimp_df_merged_pearson), meanimp_df_merged_deviance)
head(meanimp_df_residual_baseline)
meanimp_df_residual_baseline$type = 'baseline'


file = '/home/delaram/sciFA/Results/residual/pearson/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_list_shuffle_pearson <- Reduce(rbind,meanimp_list)
meanimp_list_shuffle_pearson$res = 'pearson'
head(meanimp_list_shuffle_pearson)

file = '/home/delaram/sciFA/Results/residual/response/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_shuffle_response <- Reduce(rbind,meanimp_list)
meanimp_shuffle_response$res = 'response'
head(meanimp_shuffle_response)

file = '/home/delaram/sciFA/Results/residual/deviance//'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T), read.csv)
meanimp_shuffle_deviance <- Reduce(rbind,meanimp_list)
meanimp_shuffle_deviance$res = 'deviance'
head(meanimp_shuffle_deviance)

meanimp_residual_shuffle = rbind(rbind(meanimp_list_shuffle_pearson, meanimp_shuffle_response), meanimp_shuffle_deviance)
head(meanimp_residual_shuffle)
meanimp_residual_shuffle$type = 'shuffle'


meanimp_residual_merged = rbind(meanimp_df_residual_baseline, meanimp_residual_shuffle)
head(meanimp_residual_merged)
meanimp_residual_merged_m = melt(meanimp_residual_merged)
head(meanimp_residual_merged_m)


ggplot2::ggplot(meanimp_residual_merged_m, aes(y=value, x=res, fill=type))+geom_boxplot()+
  theme_classic()+coord_flip()+ylab('Mean importance value')





meanimp_df_residual_baseline_m = melt(meanimp_df_residual_baseline)
head(meanimp_df_residual_baseline_m)
meanimp_df_residual_shuffle_m = melt(meanimp_residual_shuffle)
head(meanimp_df_residual_shuffle_m)




mean_imp_baseline_m = data.frame(cov_level=meanimp_df_residual_baseline_m$X, 
                                 factor=meanimp_df_residual_baseline_m$variable,
                                 imp_score=meanimp_df_residual_baseline_m$value,
                                 res=meanimp_df_residual_baseline_m$res)
head(mean_imp_baseline_m)



importance_df_shuffle_split <- split(meanimp_df_residual_shuffle_m, meanimp_df_residual_shuffle_m$res)
lapply(importance_df_shuffle_split, head)
importance_df_baseline_split <- split(meanimp_df_residual_baseline_m, meanimp_df_residual_baseline_m$res)
lapply(importance_df_baseline_split, head)

names(importance_df_shuffle_split)
names(importance_df_baseline_split)

for(i in 1:length(importance_df_shuffle_split)){
  a_importance_df_shuffle = importance_df_shuffle_split[[i]]
  a_importance_df_basline = importance_df_baseline_split[[i]]
  importance_df_baseline_split[[i]]$pval = sapply(1:nrow(a_importance_df_basline), function(i) 
    sum(a_importance_df_shuffle$value>a_importance_df_basline$value[i])/nrow(a_importance_df_shuffle))
}

tab=rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.05))),
          pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.01))),
          pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.001))))

gridExtra::grid.table(t(tab))
dev.off()

tab=rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.05)/180,2))),
          pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.01)/180,2))),
          pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.001)/180,2))))
gridExtra::grid.table(t(tab))

thr = 0.01
sapply(1:length(importance_df_baseline_split), function(i) 
{importance_df_baseline_split[[i]]$sig <<- importance_df_baseline_split[[i]]$pval < thr})
head(importance_df_baseline_split[[1]])

AvgFacSig_df_model = sapply(1:length(importance_df_baseline_split), function(i){
  a_model_imp_df = importance_df_baseline_split[[i]]
  a_model_imp_df_cov = split(a_model_imp_df, a_model_imp_df$X)
  AvgFacSig = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig) = names(a_model_imp_df_cov)
  return(AvgFacSig)
}, simplify = T)

colnames(AvgFacSig_df_model) = names(importance_df_baseline_split) 
head(AvgFacSig_df_model)
AvgFacSig_df_model_m = melt(AvgFacSig_df_model)
head(AvgFacSig_df_model_m)

ggplot(AvgFacSig_df_model_m, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))

