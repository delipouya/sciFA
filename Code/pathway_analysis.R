source('~/RatLiver/Codes/Functions.R')
Initialize()
library(gprofiler2)

get_gprofiler_enrich <- function(markers, model_animal_name){
  gostres <- gost(query = markers,
                  ordered_query = TRUE, exclude_iea =TRUE, 
                  sources=c('GO:BP' ,'REAC'),
                  organism = model_animal_name)
  return(gostres)
}


varimax_df = read.csv('~/sciFA/Results/varimax_loading_df_kidneyMap.csv')
head(varimax_df)

df = data.frame(gene= varimax_df$X,factor=varimax_df$F18)
model_animal_name ='hsapiens'
df_pos = df[order(df$factor, decreasing = T),]
head(df_pos,30)
num_genes = 200

enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], model_animal_name)
head(enrich_res$result,30)
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos = enrich_res_pos[1:15,]
enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'p_value')]
enrich_res_pos$log_p = -log(enrich_res_pos$p_value)
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle('Male(positive loading)')


df_neg = df[order(df$factor, decreasing = F),]
head(df_neg,30)
enrich_res = get_gprofiler_enrich(markers=df_neg$gene[1:num_genes], model_animal_name)
head(enrich_res$result,30)
enrich_res_neg = data.frame(enrich_res$result)
enrich_res_neg = enrich_res_neg[1:15,]
enrich_res_neg = enrich_res_neg[,colnames(enrich_res_neg) %in% c('term_name', 'p_value')]
enrich_res_neg$log_p = -log(enrich_res_neg$p_value)
ggplot(enrich_res_neg, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle('Female(negative loading)')


#Pathway analysis (Fig. 3a, Supplementary Data 3) revealed processes related to amino acid metabolism, 
#PT transport, and regulation of the inflammatory response as increased in females. Among the pathways increased 
#in males, processes related to mitochondrial aerobic metabolism (‘oxidative phosphorylation’, 
#‘tricarboxylic acid (TCA) cycle’ and ‘electron transport chain’) predominated. Two additional metabolic processes, 
#namely ‘generation of precursor metabolites’ and ‘nucleoside triphosphate metabolism’, were also enriched in males.

gp_enrich_res_filt <- lapply(gp_enrich_res, 
                             function(x) x$result[x$result$query_size>5 & x$result$query_size<350 & x$result$intersection_size >3, ])
names(gp_enrich_res_filt)  
lapply(gp_enrich_res_filt, head)
head(gp_enrich_res_filt[['cluster_10']], 10)

p <- gostplot(gostres, capped = T, interactive = FALSE)
# pt2 <- publish_gosttable(gostres, use_colors = TRUE, filename = NULL)







gprofiler_results_oe <- gprofiler(query = sigOE_genes, 
                                  organism = "hsapiens",
                                  ordered_query = F, 
                                  exclude_iea = F, 
                                  max_p_value = 0.05, 
                                  max_set_size = 0,
                                  correction_method = "fdr",
                                  hier_filtering = "none", 
                                  domain_size = "annotated",
                                  custom_bg = allOE_genes)
