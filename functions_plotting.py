import seaborn as sns
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd


import constants as const


plt_legend_protocol = [mpatches.Patch(color='palegreen', label='sc_10X'),
            mpatches.Patch(color='yellow', label='CELseq2'),
            mpatches.Patch(color='pink', label='Dropseq')]


plt_legend_cl = [mpatches.Patch(color='springgreen', label='HCC827'),
                        mpatches.Patch(color='red', label='H1975'),
                        mpatches.Patch(color='orchid', label='H2228')]

plt_legend_dict = {'protocol': plt_legend_protocol, 'cell_line': plt_legend_cl}




def get_colors_dict_scMix(y_protocol, y_cell_line):
    '''
    generate a dictionary of colors for each cell in the scMix dataset
    y_protocol: the protocol for each cell
    y_cell_line: the cell line for each cell
    '''

    ### generating the list of colors for samples
    my_color = {b'sc_10X': 'palegreen', b'CELseq2':'yellow', b'Dropseq':'pink'}
    ### generate a list containing the corresponding color for each sample
    protocol_color = [my_color[y_protocol[i]] for i in range(len(y_protocol))]

    my_color = {'HCC827': 'springgreen', 'H1975':'red', 'H2228':'orchid'}
    cell_line_color = [my_color[y_cell_line[i]] for i in range(len(y_cell_line))]

    return {'protocol': protocol_color, 'cell_line': cell_line_color}




def plot_pca_scMix(pca_scores, 
                   num_components_to_plot, 
                   cell_color_vec, 
                   covariate='protocol',
                   title='PCA of the data matrix') -> None:
    '''
    plot the PCA components with PC1 on the x-axis and other PCs on the y-axis
    pca_scores: the PCA scores for all the cells
    num_components_to_plot: the number of PCA components to plot as the y-axis
    cell_color_vec: the color vector for each cell
    covariate: the covariate to color the cells
    title: the title of the plot
    '''
    
    for i in range(1, num_components_to_plot):
        ## color PCA based on strain
        plt.figure()
        plt.scatter(pca_scores[:,0], pca_scores[:,i], c=cell_color_vec, s=1) 
        plt.xlabel('PC1')
        plt.ylabel('PC'+str(i+1))
        plt.title(title)
        plt.legend(handles=plt_legend_dict[covariate])
        plt.show()



def plot_umap_scMix(pca_scores, cell_color_vec , 
                    covariate='protocol', 
                    title='UMAP of the PC components of the gene expression matrix') -> None:

    ### apply UMAP to teh PCA components
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(pca_scores)
    print('embedding shape: ', embedding.shape)
    
    ### plot the UMAP embedding
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cell_color_vec, s=1)
    plt.title(title)
    plt.legend(handles=plt_legend_dict[covariate])
    
    plt.show()



def plot_fc_match_heatmap(mean_importance_df, 
                          title='Mean importance of each factor for each covariate level')  -> None:
    '''
    plot mean_importance_df as a heatmap
    mean_importance_df: dataframe of mean importance of each factor for each covariate level
    title: the title of the plot
    '''
    fig, ax = plt.subplots(figsize=(23, 4))
    ### change the column names as 'factor1', 'factor2', ...
    mean_importance_df.columns = ['F'+str(i) for i in range(1, mean_importance_df.shape[1]+1)]
    sns.heatmap(mean_importance_df, cmap='coolwarm', ax=ax, annot=True, fmt=f'.1g')
    ax.set_title(title)
    plt.show()




def plot_metric_barplot(metric_all, title) -> None:
    '''
    plot the factor metric as a bar plot
    metric_all: the metric value for all the factors
    title: the title of the plot
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(metric_all)), metric_all)
    ax.set_xticks(range(len(metric_all)))
    x_axis_label = ['F'+str(i) for i in range(1, len(metric_all)+1)]
    ax.set_xticklabels(x_axis_label, rotation=45, ha="right",)
    ax.set_title(title)
    plt.show()



def plot_factor_scatter(factor_scores, x_i, y_i, cell_color_vec, covariate='protocol',title='') -> None:
    '''
    plot the scatter plot of two factors
    factor_scores: the factor scores for all cells
    x_i: the index of the x-axis factor
    y_i: the index of the y-axis factor
    cell_color_vec: the color vector for each cell
    covariate: the covariate to color the cells
    title: the title of the plot
    '''
    plt.figure()
    plt.scatter(factor_scores[:,x_i], factor_scores[:,y_i], c=cell_color_vec, s=1) 
    plt.xlabel('F'+str(x_i+1))
    plt.ylabel('F'+str(y_i+1))
    plt.legend(handles=plt_legend_dict[covariate])
    plt.title(title)
    plt.show()



def plot_histogram(a_factor_score, title='') -> None:
    '''
    plot the histogram of a factor score
    a_factor_score: the factor score for a factor
    title: the title of the plot
    '''
    plt.hist(a_factor_score, bins=100)
    plt.xlabel('Factor score')
    plt.ylabel('count')
    plt.title(title)
    plt.show()




def plot_scaled_variance_heatmap(SV_all_factors, covariate_levels, title='Scaled variance for all the factors'):
    ''' 
    plot the scaled variance of all the factors for each covariate level
    SV_all_factors: the scaled variance of all the factors for each covariate level
    covariate_levels: the list of covariate levels
    '''
    plt.figure(figsize=(16, 3))
    plt.imshow(SV_all_factors, cmap='RdPu', interpolation='nearest')
    plt.xticks(np.arange(len(SV_all_factors[0])), np.arange(len(SV_all_factors[0]))+1)
    plt.yticks(np.arange(len(covariate_levels)), covariate_levels)
    plt.colorbar()
    plt.xlabel('Factors')
    plt.ylabel('Covariate levels')
    plt.title(title)
    plt.show()



def plot_metric_correlation(all_metrics_df) -> None:
    '''
    plot the correlation matrix for the metrics
    all_metrics_df: a pandas dataframe of the metrics for all the factors
    '''
    ### convert to a numpy array
    all_metrics_np = all_metrics_df.to_numpy()
    factor_metrics = all_metrics_df.columns

    ### calculate the correlation matrix
    corr = np.corrcoef(all_metrics_np.T)
    
    ### plot the correlation matrix with hierarchical clustering
    plt.figure(figsize=(5, 5))
    plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
    plt.xticks(np.arange(len(factor_metrics)), factor_metrics, rotation=-70)
    plt.yticks(np.arange(len(factor_metrics)), factor_metrics)
    plt.colorbar()
    plt.xlabel('factor metrics')
    plt.ylabel('factor metrics')
    plt.title('correlation matrix for factor metrics')
    plt.show()


def plot_metric_heatmap(all_metrics_scaled, factor_metrics, title='Scaled factor metrics for all factors'):
    '''
    plot the heatmap of the scaled factor metrics for all the factors
    all_metrics_scaled: the scaled factor metrics for all the factors 
    factor_metrics: the list of factor metric names
    '''
    plt.figure(figsize=(18, 3))
    plt.imshow(all_metrics_scaled.T, cmap='YlGnBu', interpolation='nearest') #'YlOrRd'
    plt.yticks(np.arange(len(factor_metrics)), factor_metrics)
    plt.xticks(np.arange(all_metrics_scaled.shape[0]), np.arange(all_metrics_scaled.shape[0])+1)
    plt.colorbar()
    plt.xlabel('factor metrics')
    plt.ylabel('factors')
    plt.title(title)


def plot_metric_heatmap_sb(all_metrics_scaled, factor_metrics, title='Scaled factor metrics for all factors'):
    '''
    plot the heatmap of the scaled factor metrics for all the factors using seaborn
    all_metrics_scaled: the scaled factor metrics for all the factors
    factor_metrics: the list of factor metric names
    '''
    df = pd.DataFrame(all_metrics_scaled.T, columns=np.arange(all_metrics_scaled.shape[0])+1)
    df.index = factor_metrics
    plt.figure(figsize=(25,4))
    sns.heatmap(df, cmap='YlGnBu', annot=True, fmt=f'.2g') #
    plt.show()



def plot_AUC_all_factors_df(AUC_all_factors_df, title=''):
    '''
    plot the AUC of all the factors for all the covariate levels
    AUC_all_factors_df: a dataframe of AUCs for all the factors
    '''
    fig, ax = plt.subplots(figsize=(20,5))
    ax = sns.heatmap(AUC_all_factors_df, cmap="YlOrBr", linewidths=.5, annot=True)
    ax.set_title(title)
    ax.set_xlabel('factor')
    ax.set_ylabel('covariate level')
    plt.show()

