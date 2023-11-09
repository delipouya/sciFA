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


importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_n800.csv')
importance_df_m_merged = read.csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded.csv')

head(importance_df_m_merged)
importance_df_m_merged
ggplot(importance_df_m_merged, aes(x=model, y=importance, fill=shuffle))+geom_boxplot()+theme_classic()+
  coord_flip()+scale_fill_manual(values=c("#999999", "maroon"))

imp_df_models<- split(importance_df_m_merged, importance_df_m_merged$model)
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


