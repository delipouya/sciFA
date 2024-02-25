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
varimax_df = read.csv('~/sciFA/Results/varimax_loading_df_lupusPBMC.csv')

head(varimax_df)

df = data.frame(gene= varimax_df$X,factor=varimax_df$F9)
model_animal_name ='hsapiens'
df_pos = df[order(df$factor, decreasing = T),]
head(df_pos,30)
num_genes = 200

table_to_vis = df_pos[1:20,]
rownames(table_to_vis) = NULL
colnames(table_to_vis) = c('Gene', 'Score')
table_to_vis$Score = round(table_to_vis$Score, 3)
#table_to_vis[2,]$Gene = 'APOBEC3A'
library(gridExtra)
dev.off()
tt2 <- ttheme_minimal()
gridExtra::grid.table(table_to_vis, theme=tt2)



library(grid)
library(gridExtra)
y<-iris[1:4,1:5]
hj <- matrix(c(0.90,  0.90, 0.90, 0.90, 0.90), ncol=ncol(y), nrow=nrow(y), byrow=TRUE)
x <- matrix(c( 0.90, 0.90, 0.90, 0.90, 0.90), ncol=ncol(y), nrow=nrow(y), byrow=TRUE)
colours <- matrix("#e9f1e9", nrow(y), ncol(y))
colours[1:nrow(colours), 1] <- "white"
tt1 <- ttheme_default(core=list(fg_params=list(hjust = as.vector(hj),
                                               x = as.vector(x), fontface=c(rep("plain",ncol(y)))),
                                bg_params = list(fill =colours , col="black")))
tab<-tableGrob(y, rows = NULL, theme = tt1)
grid.newpage()
grid.draw(tab)

enrich_res = get_gprofiler_enrich(markers=df_pos$gene[1:num_genes], model_animal_name)
head(enrich_res$result,30)
enrich_res_pos = data.frame(enrich_res$result)
enrich_res_pos$log_p = -log(enrich_res_pos$p_value)

enrich_res_pos = enrich_res_pos[order(enrich_res_pos$log_p, decreasing = T),]
#enrich_res_pos[! 1:nrow(enrich_res_pos) %in% c(7, 9, 13, 15, 17, 19),]
enrich_res_pos = enrich_res_pos[,colnames(enrich_res_pos) %in% c('term_name', 'log_p')]
(enrich_res_pos)
enrich_res_pos = enrich_res_pos[1:20,]

enrich_res_pos$term_name = gsub('metabolic process', 'metabolism',enrich_res_pos$term_name)
enrich_res_pos$term_name <- factor(enrich_res_pos$term_name, levels =  enrich_res_pos$term_name[length(enrich_res_pos$term_name):1])
#enrich_res_pos <- enrich_res_pos[-3,]
title = ''#'stim'#'Male'
enrich_res_pos = enrich_res_pos[! 1:nrow(enrich_res_pos) %in% c(5, 6, 7, 10, 11, 13, 14, 17), ]
ggplot(enrich_res_pos, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))


df_neg = df[order(df$factor, decreasing = F),]
head(df_neg,30)
enrich_res = get_gprofiler_enrich(markers=df_neg$gene[1:num_genes], model_animal_name)
head(enrich_res$result,30)
enrich_res_neg = data.frame(enrich_res$result)

enrich_res_neg = enrich_res_neg[,colnames(enrich_res_neg) %in% c('term_name', 'p_value')]
enrich_res_neg$log_p = -log(enrich_res_neg$p_value)
title = ''#'Female'

enrich_res_neg = enrich_res_neg[1:20,]
(enrich_res_neg)
enrich_res_neg$term_name = gsub('metabolic process', 'metabolism',enrich_res_neg$term_name)
enrich_res_neg$term_name <- factor(enrich_res_neg$term_name, levels =  enrich_res_neg$term_name[length(enrich_res_neg$term_name):1])

title = ''#'stim'#'Male'
ggplot(enrich_res_neg, aes(y=term_name,x=log_p))+geom_bar(stat = 'identity')+xlab('-log(p value)')+
  theme_classic()+ylab('')+ggtitle(title)+
  theme(axis.text.x = element_text(color = "grey20", size = 13, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 13, angle = 0, hjust = 1, vjust = 0, face = "plain"),  
        axis.title.x = element_text(color = "grey20", size = 17, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 17, angle = 90, hjust = .5, vjust = .5, face = "plain"))


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
