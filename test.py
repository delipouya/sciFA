

############################################################################
############################################################################
################ Don't run this cell, it's just for visualization  #########
############################################################################
############################################################################
##### Navigating individual covariate levels and their factor importance scores
### visualizing the feature importance using a heatmap
import seaborn as sns
import pandas as pd
import dataframe_image as dfi
import seaborn as sns
import pandas as pd
import dataframe_image as dfi
### visualizing the feature importance using a heatmap
import seaborn as sns
import pandas as pd
import dataframe_image as dfi


covariate_name =  'protocol'#'cell_line'
factor_scores = pca_scores
cov_level_feature_df_dict = {}

y_label = data.obs[covariate_name]

##### saving the importance of each covariate level in a dictionary
for covariate_level in y_label.unique():
    a_binary_cov = get_binary_covariate(covariate_name, covariate_level, data)
    factor_scores = pca_scores
    cov_level_feature_df_dict[covariate_level] = get_importance_df(factor_scores, a_binary_cov)


a_cov_match_factor_dict = {}
a_cov_match_factor_dict_all = {}

for covariate_level in cov_level_feature_df_dict:
    importance_df = cov_level_feature_df_dict[covariate_level]
    print(covariate_level)
    #dfi.export(importance_df.style.background_gradient(cmap='Blues', axis=1 ), '/home/delaram/scLMM/feature_heatmap_images/imp_heatmap_'+ covariate_level+'.png')
    
    ## identify the column (PC) with max value in each row (ML model)
    match_factor_table = importance_df.idxmax(axis='columns')
    highest_vote_factor = str(list(match_factor_table.mode())[0])
    print('Highest scoring factor in each model for \"'+ str(covariate_level)+ '\":\n', match_factor_table.to_frame(name=''), '\n')
    
    print('Highest factor for \"'+ str(covariate_level)+ '\" based on voting: ', str(highest_vote_factor))

    a_cov_match_factor_dict[covariate_level] = highest_vote_factor
    a_cov_match_factor_dict_all[covariate_level] = match_factor_table.to_frame(name='')


covariate_level = list(cov_level_feature_df_dict.keys())[2]
print(covariate_level)
print(highest_vote_factor)
importance_df = cov_level_feature_df_dict[covariate_level]
#importance_df.style.background_gradient(cmap='Blues', axis=1)

### visualizing the feature importance using a heatmap
importance_df_np = np.asarray(importance_df)
### plot the heatmap
fig, ax = plt.subplots(figsize=(13, 2.5))
im = ax.imshow(importance_df_np, cmap='Blues')
### add the labels
ax.set_xticks(np.arange(len(importance_df.columns)))
ax.set_yticks(np.arange(len(importance_df.index)))
ax.set_xticklabels(importance_df.columns)
ax.set_yticklabels(importance_df.index)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
### add the colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Importance', rotation=-90, va="bottom")
### show the plot
fig.tight_layout()
plt.show()

print(covariate_level)

############################################################################
import seaborn as sns
import functions_metrics as fmet


covariate_name = 'protocol'#'cell_line'
covariate_level = b'Dropseq'
factor_scores = pca_scores
covariate_vector = y_protocol

a_factor = factor_scores[:,0]
SV_level = fmet.get_scaled_variance_level(a_factor, covariate_vector, covariate_level)
SV_all = fmet.get_SV_all_levels(a_factor, covariate_vector)
a_factor_ASV_protocol = fmet.get_a_factor_ASV(a_factor, y_protocol)

ASV_protocol_all = fmet.get_ASV_all(pca_scores, y_protocol)
ASV_cell_line_all = fmet.get_ASV_all(pca_scores, y_cell_line)

print('SV_level: ', SV_level) ## scaled variance of a factor and a covariate level
print('SV_all: ', len(SV_all))
print('ASV: ', a_factor_ASV_protocol.shape)



#### plot the scaled variance for all the factors using seaborn
import seaborn as sns
df = pd.DataFrame(SV_all_factors, columns=np.arange(const.num_components)+1, index=all_covariate_levels)
plt.figure(figsize=(22, 5))
sns.heatmap(df, cmap='RdPu', annot=True, fmt='.2f')
plt.show()



###  convert SV_all_factors to a pandas dataframe
import pandas as pd
import seaborn as sns

SV_all_factors_df = pd.DataFrame(SV_all_factors, columns=np.arange(const.num_components)+1, index=all_covariate_levels)
SV_all_factors_df.style.background_gradient(cmap='Blues', axis=1)

i = 13
a_factor = pca_scores[:,i]

SV_all = fmet.get_SV_all_levels(a_factor, y_protocol)
print(y_protocol.unique())
print(SV_all)

SV_all = fmet.get_SV_all_levels(a_factor, y_cell_line)
print(y_cell_line.unique())
print(SV_all)

print(SV_all_factors.shape)
print(SV_all_factors[:,i])

print(y_protocol.unique())

print(np.unique(y_protocol))
print(np.unique(y_cell_line))
print(np.concatenate((y_protocol.unique(), y_cell_line.unique()), axis=0))
