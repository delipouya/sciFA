varimax_sc <- readRDS('~/RatLiver/Results/old_samples/varimax_rotated_OldMergedSamples_mt40_lib1500_MTremoved.rds') ## MT-removed

df = data.frame(as.matrix(varimax_sc$rotScores))
dim(df)
colnames(df) = paste0('F', 1:ncol(df))
i = 5
df$strain = unlist(lapply(strsplit(rownames(varimax_sc$rotScores), '_'), '[[', 2))
head(df)

hist(df$factor5)
hist(df$factor15)

library('MVN')

his<-MVN::mvn(data=df, subset="strain", univariatePlot="histogram")
his[1]
his[2]
his_v<-MVN::mvn(data=df[,-3])
his_v[1]
his_v[2]


library("ggpubr")
ggdensity(df$F1, xlab = "Factor scores")
ggdensity(df$F2, xlab = "Factor scores")
ggdensity(df$F3, xlab = "Factor scores")
ggdensity(df$F4, xlab = "Factor scores")
ggdensity(df$F5, xlab = "Factor scores")
ggdensity(df$F15, xlab = "Factor scores")


library(ggpubr)
ggqqplot(df$F5)


indices = runif(n = 5000, 1, length(df$F5))
shapiro.test(df$F1[indices])

factor_sub = df$F5[df$strain=='Lew']
indices = runif(n = 5000, 1, length(factor_sub))
shapiro.test(factor_sub[indices])

table(df$strain)
factor_sub = df$F5[df$strain=='DA']
indices = runif(n = 5000, 1, length(factor_sub))
shapiro.test(factor_sub[indices])




