
#######################################################
##### Human liver base model
#######################################################
gini_base_imp = list(
  'pearson'=list('AUC'=(0.42844405666813934),
                 'DecisionTree'= (0.8270269806718208),
                 'LogisticRegression'= (0.6244436026479022),
                 'XGB'= (0.6074797580935721)),
  
  'response'= list('AUC'= (0.44300881770580863),
                   'DecisionTree'= (0.8405454456704068),
                   'LogisticRegression'= (0.7181351178064663),
                   'XGB'= (0.6105563677029542)),
  
  'deviance'= list('AUC'= (0.41502764561149913),
                   'DecisionTree'= (0.8705934843417716),
                   'LogisticRegression'= (0.533822322964555),
                   'XGB'= (0.6682235875773687)))
#######################################################


gini_base_meanimp =list(
  'pearson'= list('arithmatic_minmax'= (0.43145743916258367),
                  'arithmatic_rank'= (0.21630397132616488),
                  'arithmatic_standard'= (0.3213467464296892),
                  'geometric_minmax'= (0.6900014097599303),
                  'geometric_rank'= (0.2652688750895878),
                  'geometric_standard'= (0.3446566538678779)),
  
  'response'= list('arithmatic_minmax'= (0.4482304042565286),
                   'arithmatic_rank'= (0.21807670250896058),
                   'arithmatic_standard'= (0.3348031566839708),
                   'geometric_minmax'= (0.7105488962352995),
                   'geometric_rank'= (0.26676924492481086),
                   'geometric_standard'= (0.3569829820137092)),
  
  'deviance'= list('arithmatic_minmax'= (0.4312870752553559),
                   'arithmatic_rank'= (0.2163803870967742),
                   'arithmatic_standard'= (0.33342872790755695),
                   'geometric_minmax'= (0.7297170676765146),
                   'geometric_rank'= (0.26276100746128445),
                   'geometric_standard'= (0.3607843057745439)))

#### visulize gini distributions for the human liver data
library(reshape2)
setwd('/home/delaram/sciFA/Results/benchmark_humanliver/gini_analysis/')

files_list_meanimp = list.files(pattern = 'meanimp_gini_*')
list_meanimp = lapply(files_list_meanimp, read.csv)
names(list_meanimp) = files_list_meanimp
lapply(list_meanimp, head)
list_meanimp_merged = Reduce(rbind, list_meanimp)
list_meanimp_merged = list_meanimp_merged[,-1]
head(list_meanimp_merged)
list_meanimp_merged_m = melt(list_meanimp_merged)
head(list_meanimp_merged_m)


files_list_imp = list.files(pattern = 'imp_gini_*')[1:3]
list_imp = lapply(files_list_imp, read.csv)
names(list_imp) = files_list_imp
lapply(list_imp, dim)
list_imp_merged = Reduce(rbind, list_imp)
list_imp_merged = list_imp_merged[,-1]
head(list_imp_merged)
list_imp_merged_m = melt(list_imp_merged)
head(list_imp_merged_m)

merged_all = rbind(list_imp_merged_m, list_meanimp_merged_m)

####################### formating the base datapints to be added to plot
library(data.table)
gini_base_imp_df = data.frame(rbindlist(gini_base_imp, fill=TRUE))
gini_base_imp_df$residual_type = names(gini_base_imp)
gini_base_imp_df_melt = melt(gini_base_imp_df)
gini_base_imp_df_melt

gini_base_meanimp_df = data.frame(rbindlist(gini_base_meanimp, fill=TRUE))
gini_base_meanimp_df$residual_type = names(gini_base_meanimp)
gini_base_meanimp_df_melt = melt(gini_base_meanimp_df)
gini_base_meanimp_df_melt

merged_all_base = rbind(gini_base_imp_df_melt, gini_base_meanimp_df_melt)


ggplot(list_imp_merged_m, aes(x=reorder(variable, value), y=value, fill=residual_type))+geom_boxplot()+
  theme_classic()+coord_flip()+theme(text = element_text(size=17))+xlab('')+
  scale_fill_brewer(palette = 'Set1')+ylab('Gini index')+
  geom_point(data = gini_base_imp_df_melt, color = "goldenrod2", 
             position =  position_dodge(width = .75), size = 2)


ggplot(merged_all, aes(x=reorder(variable, value), y=value, fill=residual_type))+geom_boxplot()+
  theme_classic()+theme(text = element_text(size=17))+xlab('')+coord_flip()+
  scale_fill_brewer(palette = 'Set1')+ylab('Gini index')+
  geom_point(data = merged_all_base, color = "red3", 
             position =  position_dodge(width = .75), size = 1.3)
#geom_point(aes(colour = Cell_line, shape = replicate, group = Cell_line),
#           position = position_dodge(width = .75), size = 3)
