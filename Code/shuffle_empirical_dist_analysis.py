sys.path.append('./Code/')
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from scipy.io import mmread

import ssl; ssl._create_default_https_context = ssl._create_unverified_context

import functions_processing as fproc
import constants as const
import statsmodels.api as sm
import functions_label_shuffle as flabel

np.random.seed(10)
import time


n = 500
FA_type = 'varimax'#'PCA'
scores_included = 'baseline'#'baseline'#'top_cov' 'top_FA' 

#######################################################
##### read all the csv files in './Results/importance_df_melted_scMixology_varimax_shuffle_results/' and concatenate them into one dataframe
#######################################################
importance_df_m_merged = pd.DataFrame()
for i in range(n):
    print('i: ', i)
    importance_df_m = pd.read_csv('/home/delaram/sciFA/Results/shuffle_empirical_dist/importance_df_melted_scMixology_varimax_shuffle_'+str(i)+'.csv')
    importance_df_m['shuffle'] = np.repeat('shuffle_'+str(i), importance_df_m.shape[0])
    importance_df_m_merged = pd.concat([importance_df_m_merged, importance_df_m], axis=0)
    

#### read the importance_df_melted_scMixology_varimax_baseline.csv file and concatenate it to importance_df_m_merged
importance_df_m_baseline = pd.read_csv('../Results/importance_df_melted_scMixology_varimax_baseline_n1000.csv')
importance_df_m_baseline['shuffle'] = np.repeat('baseline', importance_df_m_baseline.shape[0])
importance_df_m_merged = pd.concat([importance_df_m_merged, importance_df_m_baseline], axis=0)
### drop the Unnamed: 0 column
importance_df_m_merged.drop(columns=['Unnamed: 0'], inplace=True)

### reorder shuffle column as baseline, shuffle_0, shuffle_1, ... for visualization
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].astype('category')
importance_df_m_merged['shuffle'].cat.reorder_categories(['baseline'] + ['shuffle_'+str(i) for i in range(n)], inplace=True)
importance_df_m_merged.head()


############# replace shuffle_0, shuffle_1, ... with shuffle
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].replace(['shuffle_'+str(i) for i in range(n)], 
                                                                              ['shuffle' for i in range(n)])
importance_df_m_merged['shuffle'] = importance_df_m_merged['shuffle'].astype('category')
importance_df_m_merged['shuffle'].cat.reorder_categories(['baseline', 'shuffle'], inplace=True)


flabel.plot_importance_boxplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
flabel.plot_importance_violinplot(importance_df_m_merged, x='model', y='importance', hue='shuffle',
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')




###### sclale each model's importance score in the way that would have same min and max compared to each other
importance_df_m_merged['importance_scaled'] = importance_df_m_merged.groupby('model')['importance'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
flabel.plot_importance_boxplot(importance_df_m_merged, x='model', y='importance_scaled', hue='shuffle',
                                 title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')





############# selecting the top importance score for each model and each factor
### split the importance_df_m into dataframes based on "model" and for each 'factor' only keep the top score
importance_df_m_merged_top = importance_df_m_merged.groupby(['model', 'factor', 'shuffle']).apply(lambda x: x.nlargest(1, 'importance')).reset_index(drop=True)
### calculate the mean and sd importance of each model for baseline annd shuffle categories
importance_df_m_merged_mean = importance_df_m_merged_top.groupby(['model', 'shuffle']).mean().reset_index()
importance_df_m_merged_std = importance_df_m_merged_top.groupby(['model', 'shuffle']).std().reset_index()
### split importance_df_m_merged_mean based on model to a dictionary,with model as key and suffle and baseline as values
importance_df_m_merged_mean_dict = dict(tuple(importance_df_m_merged_mean.groupby('model')))
importance_df_m_merged_std_dict = dict(tuple(importance_df_m_merged_std.groupby('model')))

imp_drop_score_dict = {}
imp_mean_drop_dict = {}

## select the first model in importance_df_m_merged_mean_dict.keys()

for a_model in list(importance_df_m_merged_mean_dict.keys()):
    mean_l = list(importance_df_m_merged_mean_dict[a_model]['importance'])
    sd_l = list(importance_df_m_merged_std_dict[a_model]['importance'])
    numerator = mean_l[0] - mean_l[1]
    #denominator = np.sqrt(sd_l[0] * sd_l[1]) 
    denominator = sd_l[0] * sd_l[1]
    imp_mean_drop_dict[a_model] = numerator
    imp_drop_score_dict[a_model] = numerator/denominator

print(imp_mean_drop_dict)
print(imp_drop_score_dict)


### make a gourped violin plot of importance_df_m_top using sns and put the legend outside the plot
### boxplot of importance_df_m_merged using sns, shuffle is the hue, model as x axis
### put baseline as the first boxplot



############# replace shuffle_0, shuffle_1, ... with shuffle
importance_df_m_merged_top['shuffle'] = importance_df_m_merged_top['shuffle'].replace(['shuffle_'+str(i) for i in range(n)], 
                                                                              ['shuffle' for i in range(n)])
importance_df_m_merged_top['shuffle'] = importance_df_m_merged_top['shuffle'].astype('category')
importance_df_m_merged_top['shuffle'].cat.reorder_categories(['baseline', 'shuffle'], inplace=True)

flabel.plot_importance_boxplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')
flabel.plot_importance_violinplot(importance_df_m_merged_top, x='model', y='importance', hue='shuffle',xtick_fontsize=12,
                               title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')

