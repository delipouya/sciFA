import sys
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


n = 999
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
importance_df_m_baseline = pd.read_csv('/home/delaram/sciFA//Results/importance_df_melted_scMixology_varimax_baseline_n1000.csv')
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



#### save the importance_df_m_merged dataframe as a csv file
importance_df_m_merged.to_csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded.csv')




###### sclale each model's importance score in the way that would have same min and max compared to each other
importance_df_m_merged['importance_scaled'] = importance_df_m_merged.groupby('model')['importance'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
flabel.plot_importance_boxplot(importance_df_m_merged, x='model', y='importance_scaled', hue='shuffle',
                                 title='Model score comparison' + ' (' + FA_type + ' - ' + scores_included + ')')






n = 999
FA_type = 'varimax'#'PCA'
scores_included = 'baseline'#'baseline'#'top_cov' 'top_FA' 

#######################################################
##### read all the csv files in './Results/importance_df_melted_scMixology_varimax_shuffle_results/' and concatenate them into one dataframe
#######################################################
importance_df_m_merged_time = pd.DataFrame()
for i in range(n):
    print('i: ', i)
    importance_df_m = pd.read_csv('/home/delaram/sciFA/Results/shuffle_empirical_dist_time/time_df_scMixology_varimax_shuffle_'+str(i)+'.csv')
    importance_df_m['shuffle'] = np.repeat('shuffle_'+str(i), importance_df_m.shape[0])
    importance_df_m_merged_time = pd.concat([importance_df_m_merged_time, importance_df_m], axis=0)

### drop the Unnamed: 0 column
importance_df_m_merged_time.drop(columns=['Unnamed: 0', 'shuffle'], inplace=True)

importance_df_m_merged_time.head()
### create a box plot for all columns of importance_df_m_merged_time 
### fill teh color of each column with blue
### remove the grid and increase the font size of x and y ticks
### save the plot as a pdf file
importance_df_m_merged_time.boxplot(color='blue', grid=False, fontsize=12)
importance_df_m_merged_time.boxplot(grid=False, fontsize=20, rot=45, 
                                    figsize=(10, 8), color='black')
importance_df_m_merged_time.to_csv('/home/delaram/sciFA/Results/importance_df_melted_scMixology_varimax_shuffle_merged_1000allIncluded_RunTime.csv')
