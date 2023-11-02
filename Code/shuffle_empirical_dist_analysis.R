scale_minMax <- function(x){
  x_min = min(x)
  x_max = max(x)
  return (x-x_min)/(x_max-x_min)
}

importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_n800.csv')
head(importance_df_m_merged)
importance_df_m_merged
ggplot(importance_df_m_merged, aes(x=model, y=importance, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#999999", "maroon"))

library(ggpubr)
#importance_df_m_merged_base= importance_df_m_merged[importance_df_m_merged$shuffle == 'baseline',]
#
#importance_df_m_merged = importance_df_m_merged_shuffle
imp_df_models<- split(importance_df_m_merged, importance_df_m_merged_shuffle$model)
hist(imp_df_models$DecisionTree$importance)

sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_scale <<- scale(imp_df_models[[i]]$importance, center = FALSE)}, simplify = F)
sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_z_trans <<- scale(imp_df_models[[i]]$importance)}, simplify = F)
sapply(1:length(imp_df_models), function(i) {imp_df_models[[i]]$imp_minmax <<- scale_minMax(imp_df_models[[i]]$importance)}, simplify = F)
head(imp_df_models$DecisionTree)

importance_df_m_merged_shuffle_scale = Reduce(rbind, imp_df_models)
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=importance, fill=shuffle))+geom_boxplot()+
  theme_classic()+coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_z_trans, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_minmax, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))
ggplot(importance_df_m_merged_shuffle_scale, aes(x=model, y=imp_scale, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#56B4E9", "maroon"))



model_names = names(table(importance_df_m_merged_shuffle$model))
hist(model_imp_shuffle)
ggplot(importance_df_m_merged_shuffle, aes(x=importance, fill=model))+
  geom_density(alpha=0.5)+theme_classic()+scale_fill_brewer(palette = 'Set1')


get_model_imp_pvalues_df <- function(importance_df_m_merged, a_model){
  importance_df_m_merged_shuffle= importance_df_m_merged[importance_df_m_merged$shuffle == 'shuffle',]
  model_imp_shuffle_values = importance_df_m_merged_shuffle$importance[importance_df_m_merged_shuffle$model==a_model]
  
  model_imp_baseline = importance_df_m_merged[importance_df_m_merged$shuffle == 'baseline' & importance_df_m_merged$model == a_model,]
  model_imp_baseline$pvalue = sapply(1:nrow(model_imp_baseline), 
                                     function(i) sum(model_imp_shuffle_values>model_imp_baseline$importance[i])/length(model_imp_shuffle_values), 
                                     simplify = T)
  return(model_imp_baseline)
  
  
}
i = 4
a_model = model_names[i]
model_imp_shuffle_values = importance_df_m_merged_shuffle$importance[importance_df_m_merged_shuffle$model==a_model]
ggplot(importance_df_m_merged_shuffle, aes(x=importance))+geom_histogram(alpha=0.5, bins=100)+theme_classic()+ggtitle(a_model)

model_imp_pvalues_df_list = sapply(1:length(model_names), function(i){get_model_imp_pvalues_df(importance_df_m_merged, model_names[i])}, simplify = F)
names(model_imp_pvalues_df_list) = model_names
head(model_imp_pvalues_df_list$DecisionTree)
dim(model_imp_pvalues_df_list$DecisionTree)
table(model_imp_pvalues_df_list$DecisionTree$factor)

ggplot(model_imp_pvalues_df_list$DecisionTree, aes(x=pvalue))+geom_histogram(alpha=0.5, bins=100)+theme_classic()+ggtitle(a_model)

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




model_imp_pvalues_df_list_merged[model_imp_pvalues_df_list_merged$model=='']