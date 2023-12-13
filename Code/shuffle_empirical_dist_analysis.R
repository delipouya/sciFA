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

################################################################################################
########################## importance evaluation for model comparison
################################################################################################
#importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_n800.csv')
importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded.csv')
#importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/shuffle_empirical_dist_V2/importance_df_melted_scMixology_varimax_shuffle_0_V2.csv')
importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_baseline_n1000_V2.csv')
importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/')
head(importance_df_m_merged)
table(importance_df_m_merged$factor)

head(importance_df_m_merged)
table(importance_df_m_merged$factor)
ggplot(importance_df_m_merged, aes(x=model, y=importance, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#999999", "maroon"))

head(importance_df_m_merged)
imp_df_models<- split(importance_df_m_merged, importance_df_m_merged$model)
head(imp_df_models[[1]])
table(imp_df_models[[1]]$model)
table(imp_df_models[[1]]$X)
hist(imp_df_models$DecisionTree$importance)

sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_scale <<- scale(imp_df_models[[i]]$importance, center = FALSE)}, simplify = F)
sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_z_trans <<- scale(imp_df_models[[i]]$importance)}, simplify = F)
sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_minmax <<- scale_minMax(imp_df_models[[i]]$importance)}, simplify = F)
sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_max_scale <<- scale_Max(imp_df_models[[i]]$importance)}, simplify = F)

head(imp_df_models$DecisionTree)

importance_df_m_merged_shuffle_scale = Reduce(rbind, imp_df_models)
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=importance, fill=shuffle))+geom_boxplot()+
  theme_classic()+coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+
  theme(text = element_text(size=18))+xlab('')

ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_z_trans, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_minmax, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')+ylab('Importance score (min-max scaled)')



ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_scale, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_max_scale, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')






importance_df_m_merged_base= importance_df_m_merged[importance_df_m_merged$shuffle == 'baseline',]
importance_df_m_merged_shuffle = importance_df_m_merged_shuffle_scale[importance_df_m_merged_shuffle_scale$shuffle == 'shuffle',]
  
model_names = names(table(importance_df_m_merged_shuffle$model))
ggplot(importance_df_m_merged_shuffle, aes(x=importance, fill=model))+
  geom_histogram(alpha=0.5,color='black',bins=100)+theme_classic()+scale_fill_brewer(palette = 'Set1')

ggplot(importance_df_m_merged_shuffle, aes(x=imp_minmax, fill=model))+
  geom_histogram(alpha=0.5,color='black',bins=100)+theme_classic()+scale_fill_brewer(palette = 'Set1')

+geom_vline(xintercept=0.09, color = "red", size=1, linetype="dashed")

get_model_imp_pvalues_df <- function(importance_df_m_merged, a_model){
  importance_df_m_merged_shuffle= importance_df_m_merged[importance_df_m_merged$shuffle == 'shuffle',]
  model_imp_shuffle_values = importance_df_m_merged_shuffle$importance[importance_df_m_merged_shuffle$model==a_model]
  
  model_imp_baseline = importance_df_m_merged[importance_df_m_merged$shuffle == 'baseline' & importance_df_m_merged$model == a_model,]
  model_imp_baseline$pvalue = sapply(1:nrow(model_imp_baseline), 
                                     function(i) sum(model_imp_shuffle_values>model_imp_baseline$importance[i])/length(model_imp_shuffle_values), 
                                     simplify = T)
  return(model_imp_baseline)
  
  
}
model_names
i = 5
a_model = model_names[i]
a_model
model_imp_shuffle_values = importance_df_m_merged_shuffle$importance[importance_df_m_merged_shuffle$model==a_model]
ggplot(importance_df_m_merged_shuffle, aes(x=importance))+geom_histogram( bins=200,fill='grey')+
  theme_classic()+ggtitle(a_model)+theme(text = element_text(size=18))+xlab('Importance scores')+
  ylab("Frequency")+geom_vline(xintercept=0.09, color = "red", size=1, linetype="dashed")

max(importance_df_m_merged_shuffle$importance)


model_imp_pvalues_df_list = sapply(1:length(model_names), function(i){get_model_imp_pvalues_df(importance_df_m_merged, model_names[i])}, simplify = F)
names(model_imp_pvalues_df_list) = model_names
head(model_imp_pvalues_df_list$DecisionTree)
dim(model_imp_pvalues_df_list$DecisionTree)
table(model_imp_pvalues_df_list$DecisionTree$factor)

ggplot(model_imp_pvalues_df_list$DecisionTree, aes(x=pvalue))+geom_histogram(alpha=0.8, bins=100)+theme_classic()+ggtitle(a_model)

model_imp_pvalues_df_list_merged = Reduce(rbind, model_imp_pvalues_df_list)
head(model_imp_pvalues_df_list_merged)

ggplot(model_imp_pvalues_df_list_merged, aes(x=model, y=pvalue, fill=model))+geom_boxplot(alpha=0.7)+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+coord_flip()
ggplot(model_imp_pvalues_df_list_merged, aes(x=model, y=pvalue, fill=model))+geom_violin(alpha=0.7)+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+coord_flip()

ggplot(model_imp_pvalues_df_list_merged, aes(x=pvalue, fill=model))+
  geom_density(alpha=0.5)+theme_classic()+scale_fill_brewer(palette = 'Set1')

ggplot(model_imp_pvalues_df_list_merged, aes(x=pvalue, y=importance,color=model))+
  geom_point(alpha=0.5)+theme_classic()+scale_color_brewer(palette = 'Set1')




sum(model_imp_pvalues_df_list_merged$pvalue[model_imp_pvalues_df_list_merged$model=='XGB'] < 0.05)
sum(model_imp_pvalues_df_list_merged$pvalue[model_imp_pvalues_df_list_merged$model=='RandomForest'] < 0.05)
sum(model_imp_pvalues_df_list_merged$pvalue[model_imp_pvalues_df_list_merged$model=='DecisionTree'] < 0.05)


table(importance_df_m_merged$shuffle) 
importance_df_m_merged_base = importance_df_m_merged[importance_df_m_merged$shuffle=='baseline',]
head(importance_df_m_merged_base)
cor_df = data.frame(imp=importance_df_m_merged_base$importance, model=importance_df_m_merged_base$model)
cor_df_models<- split(cor_df, cor_df$model)
lapply(cor_df_models, head)
sapply(1:length(cor_df_models), function(i) colnames(cor_df_models[[i]])[1]<<-names(cor_df_models)[i])
lapply(cor_df_models, head)
cor_df_merged = Reduce(cbind, cor_df_models)
head(cor_df_merged)
cor_df_merged <- cor_df_merged[,colnames(cor_df_merged) %in% names(cor_df_models)]
head(cor_df_merged)
cor_mat = cor(cor_df_merged)
colnames(cor_mat) = c( "DecisionTree","KNNpermute", "LogisticReg", "RandomForest","XGB"  )
rownames(cor_mat) = c( "DecisionTree","KNNpermute", "LogisticReg", "RandomForest","XGB"  )
pheatmap::pheatmap(cor_mat, display_numbers = TRUE)


################################################################################################
########################## mean importance evaluation for mean and scaling comparison
################################################################################################
file = '/home/delaram/sciFA/Results/shuffle_empirical_dist_V2/mean_imp/'
file = '/home/delaram/sciFA/Results/shuffle_empirical_dist_V3/mean_imp/'

mean_imp_list = lapply(list.files(file, full.names = T), read.csv)
length(mean_imp_list)
mean_imp_shuffle <- Reduce(rbind,mean_imp_list)
head(mean_imp_shuffle)
mean_imp_baseline <- read.csv('/home/delaram/sciFA/Results/meanimp_df_scMixology_varimax_baseline_V3.csv')
head(mean_imp_baseline)

colnames(mean_imp_shuffle) == colnames(mean_imp_baseline)
mean_imp_df = rbind(mean_imp_baseline, mean_imp_shuffle)
head(mean_imp_df)
mean_imp_df$scale_mean = paste0(mean_imp_df$scale_type, '_', mean_imp_df$mean_type)
table(mean_imp_df$scale_mean)
mean_imp_df_split = split(mean_imp_df, mean_imp_df$scale_mean)
lapply(mean_imp_df_split, head)

names(mean_imp_df_split)

get_melted_meanDF <- function(a_df){
  a_df = a_df[,!colnames(a_df) %in%c('X', 'covariate')]
  a_df_m = melt(a_df, id.vars =c('scale_mean','scores_included'), measure.vars = paste0('F',1:30),value.name = c('factor'))
  return(a_df_m)
}
mean_imp_melted_list = lapply(mean_imp_df_split, get_melted_meanDF)
mean_imp_melted_list_2 = mean_imp_melted_list
for(i in 1:length(mean_imp_melted_list)){
 
  mean_imp_melted_list_2[[i]]$factor_max = scale_Max( mean_imp_melted_list_2[[i]]$factor)
  mean_imp_melted_list_2[[i]]$factor_minmax = scale_minMax( mean_imp_melted_list_2[[i]]$factor)
  
}
mean_imp_melted = Reduce(rbind, mean_imp_melted_list_2)

head(mean_imp_melted)
mean_imp_melted$factor = as.numeric(mean_imp_melted$factor)
summary(mean_imp_melted$factor)
table(mean_imp_melted$scores_included)
table(mean_imp_melted$scale_mean)
head(mean_imp_melted[mean_imp_melted$scale_mean=='standard_geometric',])
table(mean_imp_melted$scale_mean[is.na(mean_imp_melted$factor)])

#mean_imp_melted = mean_imp_melted[!mean_imp_melted$scale_mean %in% c('rank_arithmatic','rank_geometric'),]
ggplot2::ggplot(mean_imp_melted, aes(x=scale_mean, y=factor, fill=scores_included))+geom_boxplot()+coord_flip()+theme_classic()+
  scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')
ggplot2::ggplot(mean_imp_melted, aes(x=scale_mean, y=factor_max, fill=scores_included))+geom_boxplot()+coord_flip()+theme_classic()+
  scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')
ggplot2::ggplot(mean_imp_melted, aes(x=scale_mean, y=factor_minmax, fill=scores_included))+geom_boxplot()+coord_flip()+theme_classic()+
  scale_fill_manual(values=c("#56B4E9", "maroon"))+theme(text = element_text(size=18))+xlab('')






table(mean_imp_melted$scale_mean)
summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='minmax_arithmatic' & mean_imp_melted$scores_included=='baseline'])
summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='minmax_arithmatic' & mean_imp_melted$scores_included=='shuffle'])

summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='minmax_geometric' & mean_imp_melted$scores_included=='baseline'])
summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='minmax_geometric' & mean_imp_melted$scores_included=='shuffle'])


summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='rank_arithmatic' & mean_imp_melted$scores_included=='baseline'])
summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='rank_arithmatic' & mean_imp_melted$scores_included=='shuffle'])


summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='rank_geometric' & mean_imp_melted$scores_included=='baseline'])
summary(mean_imp_melted$factor[mean_imp_melted$scale_mean=='rank_geometric' & mean_imp_melted$scores_included=='shuffle'])




###############################################################################################
############################## Refining the representation of shuffling results ##############################
##########################################################################################
############################################################
importance_df_m_merged_shuffle = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded.csv')
importance_df_m_merged_baseline = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_baseline_n1000.csv')
head(importance_df_m_merged_shuffle)
dim(importance_df_m_merged_baseline)

importance_df_shuffle_split <- split(importance_df_m_merged_shuffle, importance_df_m_merged_shuffle$model)
lapply(importance_df_shuffle_split, head)
importance_df_baseline_split <- split(importance_df_m_merged_baseline, importance_df_m_merged_baseline$model)
lapply(importance_df_baseline_split, head)

for(i in 1:length(importance_df_shuffle_split)){
  a_importance_df_shuffle = importance_df_shuffle_split[[i]]
  a_importance_df_basline = importance_df_baseline_split[[i]]
  importance_df_baseline_split[[i]]$pval = sapply(1:nrow(a_importance_df_basline), function(i) 
    sum(a_importance_df_shuffle$importance>a_importance_df_basline$importance[i])/nrow(a_importance_df_shuffle))
}

gridExtra::grid.table(rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.05))),
      pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.01))),
      pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.001)))))
dev.off()

gridExtra::grid.table(rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.05)/180,2))),
                            pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.01)/180,2))),
                            pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.001)/180,2)))))

thr = 0.001
sapply(1:length(importance_df_baseline_split), function(i) 
  {importance_df_baseline_split[[i]]$sig <<- importance_df_baseline_split[[i]]$pval < thr})
head(importance_df_baseline_split[[1]])

AvgFacSig_df_model = sapply(1:length(importance_df_baseline_split), function(i){
  a_model_imp_df = importance_df_baseline_split[[i]]
  a_model_imp_df_cov = split(a_model_imp_df, a_model_imp_df$covariate_level)
  AvgFacSig = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig) = names(a_model_imp_df_cov)
  return(AvgFacSig)
}, simplify = T)

colnames(AvgFacSig_df_model) = names(importance_df_baseline_split) 
head(AvgFacSig_df_model)
AvgFacSig_df_model_m = melt(AvgFacSig_df_model)
ggplot(AvgFacSig_df_model_m, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))

#################################################################

#################################################################
importance_df_m_merged_shuffle = read.csv('/home/delaram/sciFA/Results/shuffle_empirical_dist_V2/')
importance_df_m_merged_baseline = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_baseline_n1000.csv')
head(importance_df_m_merged_shuffle)
dim(importance_df_m_merged_baseline)

importance_df_shuffle_split <- split(importance_df_m_merged_shuffle, importance_df_m_merged_shuffle$model)
lapply(importance_df_shuffle_split, head)
importance_df_baseline_split <- split(importance_df_m_merged_baseline, importance_df_m_merged_baseline$model)
lapply(importance_df_baseline_split, head)

for(i in 1:length(importance_df_shuffle_split)){
  a_importance_df_shuffle = importance_df_shuffle_split[[i]]
  a_importance_df_basline = importance_df_baseline_split[[i]]
  importance_df_baseline_split[[i]]$pval = sapply(1:nrow(a_importance_df_basline), function(i) 
    sum(a_importance_df_shuffle$importance>a_importance_df_basline$importance[i])/nrow(a_importance_df_shuffle))
}

gridExtra::grid.table(rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.05))),
                            pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.01))),
                            pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) sum(x$pval<0.001)))))
dev.off()

gridExtra::grid.table(rbind(pval_0.05=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.05)/180,2))),
                            pval_0.01=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.01)/180,2))),
                            pval_0.001=data.frame(lapply(importance_df_baseline_split, function(x) round(sum(x$pval<0.001)/180,2)))))

thr = 0.001
sapply(1:length(importance_df_baseline_split), function(i) 
{importance_df_baseline_split[[i]]$sig <<- importance_df_baseline_split[[i]]$pval < thr})
head(importance_df_baseline_split[[1]])

AvgFacSig_df_model = sapply(1:length(importance_df_baseline_split), function(i){
  a_model_imp_df = importance_df_baseline_split[[i]]
  a_model_imp_df_cov = split(a_model_imp_df, a_model_imp_df$covariate_level)
  AvgFacSig = sapply(1:length(a_model_imp_df_cov), function(i){
    sum(a_model_imp_df_cov[[i]]$sig)
  })
  names(AvgFacSig) = names(a_model_imp_df_cov)
  return(AvgFacSig)
}, simplify = T)

colnames(AvgFacSig_df_model) = names(importance_df_baseline_split) 
head(AvgFacSig_df_model)
AvgFacSig_df_model_m = melt(AvgFacSig_df_model)
dim(AvgFacSig_df_model_m)
ggplot(AvgFacSig_df_model_m, aes(y=value,x=Var2))+geom_boxplot()+
  theme_classic()+scale_fill_brewer(palette = 'Set1')+
  coord_flip()+theme(text = element_text(size=17))+xlab('')+
  ylab('Average #sig matched factors per covariate level')+
  geom_hline(yintercept=1, color = "red", size=1, linetype="dashed")+
  ggtitle(paste0('pvalue threshold=',thr))

#################################################################

################################################################################################
########################## mean importance evaluation for mean and scaling comparison
################################################################################################
file = '/home/delaram/sciFA/Results/shuffle_empirical_dist_V3/mean_imp/'

mean_imp_list = lapply(list.files(file, full.names = T), read.csv)
length(mean_imp_list)
mean_imp_shuffle <- Reduce(rbind,mean_imp_list)
head(mean_imp_shuffle)
mean_imp_shuffle_m = melt(mean_imp_shuffle)
mean_imp_shuffle_m$scale_mean = paste0(mean_imp_shuffle_m$scale_type, '_', mean_imp_shuffle_m$mean_type)
table(mean_imp_shuffle_m$scale_mean)
mean_imp_shuffle_m = data.frame(cov_level=mean_imp_shuffle_m$X, 
                                 factor=mean_imp_shuffle_m$variable,
                                 imp_score=mean_imp_shuffle_m$value,
                                 scale_mean=mean_imp_shuffle_m$scale_mean)
head(mean_imp_shuffle_m)


mean_imp_baseline <- read.csv('/home/delaram/sciFA/Results/meanimp_df_scMixology_varimax_baseline_V3.csv')
head(mean_imp_baseline)
mean_imp_baseline_m = melt(mean_imp_baseline)
mean_imp_baseline_m$scale_mean = paste0(mean_imp_baseline_m$scale_type, '_', mean_imp_baseline_m$mean_type)
head(mean_imp_baseline_m)
mean_imp_baseline_m = data.frame(cov_level=mean_imp_baseline_m$X, 
                                 factor=mean_imp_baseline_m$variable,
                                 imp_score=mean_imp_baseline_m$value,
                                 scale_mean=mean_imp_baseline_m$scale_mean)
head(mean_imp_baseline_m)



importance_df_shuffle_split <- split(mean_imp_shuffle_m, mean_imp_shuffle_m$scale_mean)
lapply(importance_df_shuffle_split, head)
importance_df_baseline_split <- split(mean_imp_baseline_m, mean_imp_baseline_m$scale_mean)
lapply(importance_df_baseline_split, head)

for(i in 1:length(importance_df_shuffle_split)){
  a_importance_df_shuffle = importance_df_shuffle_split[[i]]
  a_importance_df_basline = importance_df_baseline_split[[i]]
  importance_df_baseline_split[[i]]$pval = sapply(1:nrow(a_importance_df_basline), function(i) 
    sum(a_importance_df_shuffle$imp_score>a_importance_df_basline$imp_score[i])/nrow(a_importance_df_shuffle))
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
  a_model_imp_df_cov = split(a_model_imp_df, a_model_imp_df$cov_level)
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

#################################################################
