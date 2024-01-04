library(ggplot2)
library(ggpubr)
library(reshape2)
library(ggplot2)

###############################################################################################
########################## importance evaluation for model comparison
################################################################################################

importance_df_m_merged_pearson = read.csv('/home/delaram/sciFA/Results/benchmark/pearson/base/importance_df_melted_scMixology_pearson_baseline.csv')
importance_df_m_merged_pearson$res = 'pearson'

importance_df_m_merged_response = read.csv('/home/delaram/sciFA/Results/benchmark/response/base/importance_df_melted_scMixology_response_baseline.csv')
importance_df_m_merged_response$res = 'response'
dim(importance_df_m_merged_response)


importance_df_m_merged_deviance = read.csv('/home/delaram/sciFA/Results/benchmark/deviance//base/importance_df_melted_scMixology_deviance_baseline.csv')
importance_df_m_merged_deviance$res = 'deviance'
dim(importance_df_m_merged_deviance)

importance_df_residual_baseline = rbind(rbind(importance_df_m_merged_response, importance_df_m_merged_pearson),importance_df_m_merged_deviance) #
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
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T)[1:500], read.csv)
imp_shuffle_deviance <- Reduce(rbind,imp_list)
imp_shuffle_deviance$res = 'deviance'
head(imp_shuffle_deviance)
dim(imp_shuffle_deviance)

importance_df_residual_shuffle = rbind(rbind(imp_shuffle_pearson, imp_shuffle_response),imp_shuffle_deviance) #
importance_df_residual_shuffle$type = 'shuffle'
head(importance_df_residual_shuffle)



################## Some visualization of the results ################## 
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
########################################################################

########## splitting the baseline data frames
table(importance_df_residual_baseline$residual_type)
table(importance_df_residual_baseline$model)

residual_type_names = names(table(importance_df_residual_baseline$residual_type))
model_type_names = names(table(importance_df_residual_baseline$model))

for (residual_type in residual_type_names){
  res_df = importance_df_residual_baseline[importance_df_residual_baseline$residual_type==residual_type,]
  for (model_type in model_type_names){
    res_df_model = res_df[res_df$model==model_type,]
    print(head(res_df_model, 3))
    print(paste0(residual_type, '_', model_type))
    write.csv(res_df_model,paste0('~/sciFA/Results/benchmark/analysis/imp/baseline_df_',residual_type, '_', model_type,'.csv'))
  }
}

########################################
########## splitting the shuffled data frames

table(importance_df_residual_shuffle$residual_type)
table(importance_df_residual_shuffle$model)

residual_type_names = names(table(importance_df_residual_shuffle$residual_type))
model_type_names = names(table(importance_df_residual_shuffle$model))

for (residual_type in residual_type_names){
  res_df = importance_df_residual_shuffle[importance_df_residual_shuffle$residual_type==residual_type,]
  for (model_type in model_type_names){
    res_df_model = res_df[res_df$model==model_type,]
    print(head(res_df_model, 3))
    print(paste0(residual_type, '_', model_type))
    write.csv(res_df_model,paste0('~/sciFA/Results/benchmark/analysis/imp/shuffle_df_',residual_type, '_', model_type,'.csv'))
  }
}
########################################

thr = 0.05
cov_level_names =c("b'CELseq2'", "b'Dropseq'", "b'sc_10X'", "H1975","H2228","HCC827" )
summary_df = data.frame(matrix(nrow = length(residual_type_names)*length(model_type_names),ncol = length(cov_level_names)))
colnames(summary_df) = cov_level_names

row=1
for (residual_type in residual_type_names){
  for (model_type in model_type_names){

    basline_df = read.csv(paste0('~/sciFA/Results/benchmark/analysis/imp/baseline_df_',residual_type, '_', model_type,'.csv'))
    shuffle_df = read.csv(paste0('~/sciFA/Results/benchmark/analysis/imp/shuffle_df_',residual_type, '_', model_type,'.csv'))
    
    ### calculating pvalue for each data point in  
    basline_df$pval = sapply(1:nrow(basline_df), function(i) 
      sum(shuffle_df$importance>basline_df$importance[i])/nrow(shuffle_df))
    
    row_name = paste0(residual_type, '-', model_type)
    print(paste0(rep('----',20),collapse = ''))
    print(row_name)
    print((head(basline_df, 3)))
    print(summary(basline_df$pval))
    
    #### defining which elements in baseline are considered as significant based on the threshold
    sapply(1:length(basline_df), function(i) {basline_df$sig <<- basline_df$pval < thr})
    
    a_model_imp_df_cov = split(basline_df, basline_df$covariate_level)
    AvgFacSig_per_cov = sapply(1:length(a_model_imp_df_cov), function(i){
      sum(a_model_imp_df_cov[[i]]$sig)
    })
    names(AvgFacSig_per_cov) = names(a_model_imp_df_cov)
    summary_df[row,]=AvgFacSig_per_cov
    rownames(summary_df)[row]=row_name
    row=row+1
    
      
  }
}


summary_df_m_imp = melt( t(summary_df))
head(summary_df_m_imp)
ggplot(summary_df_m_imp, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))




###############################################################################################
########################## importance evaluation for model comparison
################################################################################################

meanimp_df_merged_pearson = read.csv('~/sciFA/Results/benchmark/pearson/base/meanimp_df_scMixology_pearson_baseline.csv')
meanimp_df_merged_pearson$res = 'pearson'
meanimp_df_merged_response = read.csv('~/sciFA/Results/benchmark/response/base/meanimp_df_scMixology_response_baseline.csv')
meanimp_df_merged_response$res = 'response'
meanimp_df_merged_deviance = read.csv('~/sciFA/Results/benchmark/deviance/base/meanimp_df_scMixology_deviance_baseline.csv')
meanimp_df_merged_deviance$res = 'deviance'

meanimp_df_residual_baseline = rbind(rbind(meanimp_df_merged_response, meanimp_df_merged_pearson),meanimp_df_merged_deviance) #
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

meanimp_residual_shuffle = rbind(rbind(meanimp_list_shuffle_pearson, meanimp_shuffle_response),meanimp_shuffle_deviance)#
head(meanimp_residual_shuffle)
meanimp_residual_shuffle$type = 'shuffle'


meanimp_residual_merged = rbind(meanimp_df_residual_baseline, meanimp_residual_shuffle)
head(meanimp_residual_merged)
meanimp_residual_merged_m = melt(meanimp_residual_merged)
head(meanimp_residual_merged_m)

ggplot2::ggplot(meanimp_residual_merged_m, aes(y=value, x=res, fill=type))+geom_boxplot()+
  theme_classic()+coord_flip()+ylab('Mean importance value')

###################### Comparing mean importance benchmark - residuals,mean,scale comparison 

meanimp_df_residual_baseline_m = melt(meanimp_df_residual_baseline)
meanimp_df_residual_shuffle_m = melt(meanimp_residual_shuffle)
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
      write.csv(res_df_mean_scale,paste0('~/sciFA/Results/benchmark/analysis/meanimp/baseline_df_',residual_type, '_', mean_type, '_', scale_type,'.csv'))
    }
  }
}


########## splitting the shuffle dataframes
head(meanimp_df_residual_shuffle_m)

table(meanimp_df_residual_shuffle_m$residual_type)
table(meanimp_df_residual_shuffle_m$mean_type)
table(meanimp_df_residual_shuffle_m$scale_type)


for (residual_type in residual_type_names){
  res_df = meanimp_df_residual_shuffle_m[meanimp_df_residual_shuffle_m$residual_type==residual_type,]
  for (mean_type in mean_type_names){
    res_df_mean = res_df[res_df$mean_type==mean_type,]
    for (scale_type in scale_type_names){
      res_df_mean_scale = res_df_mean[res_df_mean$scale_type==scale_type,]
      print(dim(res_df_mean_scale))
      print(head(res_df_mean_scale, 3))
      print(paste0(residual_type, '_', mean_type, '_', scale_type,'.csv'))
      write.csv(res_df_mean_scale,paste0('~/sciFA/Results/benchmark/analysis/meanimp/shuffle_df_',residual_type, '_', mean_type, '_', scale_type,'.csv'))
    }
  }
}



thr = 0.05
summary_df = data.frame(matrix(nrow = length(residual_type_names)*length(mean_type_names)*length(scale_type_names),
                  ncol = 6))
colnames(summary_df) = c("b'CELseq2'", "b'Dropseq'", "b'sc_10X'",  "H1975","H2228","HCC827" )

row=1
for (residual_type in residual_type_names){
  for (mean_type in mean_type_names){
    for (scale_type in scale_type_names){
      basline_df = read.csv(paste0('~/sciFA/Results/benchmark/analysis/meanimp/baseline_df_',
                                   residual_type, '_', mean_type, '_', scale_type,'.csv'))
      shuffle_df = read.csv(paste0('~/sciFA/Results/benchmark/analysis/meanimp/shuffle_df_',
                                   residual_type, '_', mean_type, '_', scale_type,'.csv'))
      ### calculating pvalue for each data point in  
      basline_df$pval = sapply(1:nrow(basline_df), function(i) 
        sum(shuffle_df$value>basline_df$value[i])/nrow(shuffle_df))
      
      row_name = paste0(residual_type, '-', mean_type, '-', scale_type)
      print(paste0(rep('----',20),collapse = ''))
      print(row_name)
      print((head(basline_df, 3)))
      print(summary(basline_df$pval))
      
      #### defining which elements in baseline are considered as significant based on the threshold
      sapply(1:length(basline_df), function(i) {basline_df$sig <<- basline_df$pval < thr})
      
      a_model_imp_df_cov = split(basline_df, basline_df$X)
      AvgFacSig_per_cov = sapply(1:length(a_model_imp_df_cov), function(i){
          sum(a_model_imp_df_cov[[i]]$sig)
        })
      names(AvgFacSig_per_cov) = names(a_model_imp_df_cov)
      summary_df[row,]=AvgFacSig_per_cov
      rownames(summary_df)[row]=row_name
      row=row+1
      

    }
  }
}


summary_df_m_meanimp = melt( t(summary_df))
ggplot(summary_df_m_meanimp, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))



summary_df_m_both = rbind(summary_df_m_meanimp, summary_df_m_imp)
ggplot(summary_df_m_both, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))


summary_df_m_both$cov=ifelse(summary_df_m_both$Var1 %in% c('H1975', 'H2228', 'HCC827'), 'cell', 'protocol')
table(summary_df_m_both$cov)

ggplot(summary_df_m_both, aes(y=value,x=Var2, color=cov))+geom_point()+
  theme_classic()+
  coord_flip()+theme(text = element_text(size=15))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+ggtitle(paste0('pvalue threshold=',thr))
  #geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  










library(reshape2)

#######################################################################
file = '/home/delaram/sciFA/Results/benchmark/pearson/shuffle/imp//'
imp_list = lapply(list.files(file, pattern = "importance_df*", full.names = T), read.csv)
imp_shuffle_pearson <- Reduce(rbind,imp_list)
imp_shuffle_pearson$res = 'pearson'
head(imp_shuffle_pearson)
dim(imp_shuffle_pearson)
imp_shuffle_pearson_model = split(imp_shuffle_pearson,imp_shuffle_pearson$model)

imp_shuffle_pearson_a_model = imp_shuffle_pearson_model[[1]]
head(imp_shuffle_pearson_a_model)
imp_shuffle_pearson_a_model = imp_shuffle_pearson_a_model[,c('factor', 'importance', 'covariate_level')]
imp_shuffle_pearson_a_model
reshape(imp_shuffle_pearson_a_model, idvar = "covariate_level", timevar = "factor", direction = "wide")







