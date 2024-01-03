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

importance_df_m_merged_pearson = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/base/importance_df_melted_scMixology_pearson_baseline.csv')
importance_df_m_merged_pearson$res = 'pearson'

importance_df_m_merged_response = read.csv('/home/delaram/sciFA/Results/benchmark/response/base/importance_df_melted_scMixology_response_baseline.csv')
importance_df_m_merged_response$res = 'response'
dim(importance_df_m_merged_response)

importance_df_residual_baseline = rbind(rbind(importance_df_m_merged_response, importance_df_m_merged_pearson)) #importance_df_m_merged_deviance
head(importance_df_residual_baseline)
importance_df_residual_baseline$type = 'baseline'
dim(importance_df_residual_baseline)

file = '/home/delaram/sciFA/Results/benchmark/pearson/shuffle/imp//'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T)[1:500], read.csv)
imp_shuffle_pearson <- Reduce(rbind,imp_list)
imp_shuffle_pearson$res = 'pearson'
head(imp_shuffle_pearson)

file = '/home/delaram/sciFA/Results/benchmark/response//shuffle/imp//'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T)[1:500], read.csv)
imp_shuffle_response <- Reduce(rbind,imp_list)
imp_shuffle_response$res = 'response'
head(imp_shuffle_response)

file = '/home/delaram/sciFA/Results/benchmark/deviance/shuffle/imp//'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T), read.csv)
imp_shuffle_deviance <- Reduce(rbind,imp_list)
imp_shuffle_deviance$res = 'deviance'
head(imp_shuffle_deviance)
dim(imp_shuffle_deviance)

importance_df_residual_shuffle = rbind(rbind(imp_shuffle_pearson, imp_shuffle_response)) #imp_shuffle_deviance
head(importance_df_residual_shuffle)
importance_df_residual_shuffle$type = 'shuffle'

library(ggplot2)
importance_df_residual_merged = rbind(importance_df_residual_baseline, importance_df_residual_shuffle)
importance_df_residual_merged$importance_abs = abs(importance_df_residual_merged$importance)
ggplot(importance_df_residual_merged, aes(y=importance_abs, x=model, fill=type))+geom_boxplot()+theme_classic()+
  coord_flip()+ggtitle('model comparison pearson and response combined')

ggplot(importance_df_residual_merged[importance_df_residual_merged$residual_type=='pearson',], 
       aes(y=importance_abs, x=model, fill=type))+geom_boxplot()+theme_classic()+
  coord_flip()+ggtitle('model comparison pearson ')

ggplot(importance_df_residual_merged[importance_df_residual_merged$residual_type=='response',], 
       aes(y=importance_abs, x=model, fill=type))+geom_boxplot()+theme_classic()+
  coord_flip()+ggtitle('model comparison response ')

ggplot(importance_df_residual_merged, aes(y=importance_abs, x=res, fill=type))+geom_boxplot()+theme_classic()+coord_flip()

head(importance_df_residual_merged)

###############################################################################################
########################## importance evaluation for model comparison
################################################################################################

meanimp_df_merged_pearson = read.csv('~/sciFA/Results/benchmark/pearson/base/meanimp_df_scMixology_pearson_baseline.csv')
meanimp_df_merged_pearson$res = 'pearson'
meanimp_df_merged_response = read.csv('~/sciFA/Results/benchmark/response///base/meanimp_df_scMixology_response_baseline.csv')
meanimp_df_merged_response$res = 'response'
meanimp_df_merged_deviance = read.csv('~/sciFA/Results/benchmark/deviance//base/meanimp_df_scMixology_deviance_baseline.csv')
meanimp_df_merged_deviance$res = 'deviance'

meanimp_df_residual_baseline = rbind(rbind(meanimp_df_merged_response, meanimp_df_merged_pearson))#meanimp_df_merged_deviance
head(meanimp_df_residual_baseline)
meanimp_df_residual_baseline$type = 'baseline'


file = '/home/delaram/sciFA/Results/benchmark/pearson/shuffle/meanimp/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T)[1:500], read.csv)
meanimp_list_shuffle_pearson <- Reduce(rbind,meanimp_list)
meanimp_list_shuffle_pearson$res = 'pearson'
head(meanimp_list_shuffle_pearson)

file = '/home/delaram/sciFA/Results/benchmark/response//shuffle/meanimp/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T)[1:500], read.csv)
meanimp_shuffle_response <- Reduce(rbind,meanimp_list)
meanimp_shuffle_response$res = 'response'
head(meanimp_shuffle_response)

file = '/home/delaram/sciFA/Results/benchmark/deviance//shuffle/meanimp/'
meanimp_list = lapply(list.files(file, pattern = "meanimp*", full.names = T)[1:500], read.csv)
meanimp_shuffle_deviance <- Reduce(rbind,meanimp_list)
meanimp_shuffle_deviance$res = 'deviance'
head(meanimp_shuffle_deviance)

meanimp_residual_shuffle = rbind(rbind(meanimp_list_shuffle_pearson, meanimp_shuffle_response))#meanimp_shuffle_deviance
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



###################### Comparing residuals - ALL included
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


###################### Comparing residuals - ALL included

head(meanimp_df_residual_baseline_m)
head(meanimp_df_residual_shuffle_m)

########## splitting the baseline dataframes
table(meanimp_df_residual_baseline_m$residual_type)
table(meanimp_df_residual_baseline_m$mean_type)
table(meanimp_df_residual_baseline_m$scale_type)

residual_type_names = names(table(meanimp_df_residual_baseline_m$residual_type))
mean_type_names = names(table(meanimp_df_residual_baseline_m$mean_type))
scale_type_names = names(table(meanimp_df_residual_baseline_m$scale_type))
 
for (residual_type in residual_type_names){
  res_df = meanimp_df_residual_baseline_m[meanimp_df_residual_baseline_m$residual_type==residual_type,]
  for (mean_type in mean_type_names){
    res_df_mean = res_df[res_df$mean_type==mean_type,]
    for (scale_type in scale_type_names){
      res_df_mean_scale = res_df_mean[res_df_mean$scale_type==scale_type,]
      print(head(res_df_mean_scale, 20))
      print(paste0(residual_type, '_', mean_type, '_', scale_type,'.csv'))
    }
  }
}


########## splitting the shuffle dataframes
head(meanimp_df_residual_shuffle_m)

table(meanimp_df_residual_shuffle_m$residual_type)
table(meanimp_df_residual_shuffle_m$mean_type)
table(meanimp_df_residual_shuffle_m$scale_type)

residual_type_names = names(table(meanimp_df_residual_shuffle_m$residual_type))
mean_type_names = names(table(meanimp_df_residual_shuffle_m$mean_type))
scale_type_names = names(table(meanimp_df_residual_shuffle_m$scale_type))

for (residual_type in residual_type_names){
  res_df = meanimp_df_residual_shuffle_m[meanimp_df_residual_shuffle_m$residual_type==residual_type,]
  for (mean_type in mean_type_names){
    res_df_mean = res_df[res_df$mean_type==mean_type,]
    for (scale_type in scale_type_names){
      res_df_mean_scale = res_df_mean[res_df_mean$scale_type==scale_type,]
      print(dim(res_df_mean_scale))
      print(head(res_df_mean_scale, 3))
      print(paste0(residual_type, '_', mean_type, '_', scale_type,'.csv'))
    }
  }
}



