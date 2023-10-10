import seaborn as sns
import numpy as np
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
sns.set(color_codes=True)
import random
random.seed(0)
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



def get_colors_dict_ratLiver(y_sample, y_strain,y_cluster):
    '''
    generate a dictionary of colors for each cell in the rat liver dataset
    y_sample: the sample for each cell
    y_strain: the strain for each cell
    y_cluster: the cluster for each cell
    '''

    my_color = {'DA_01': 'red','DA_02': 'orange', 'LEW_01': 'blue', 'LEW_02': 'purple'}
    sample_color = [my_color[y_sample[i]] for i in range(len(y_sample))]

    ### make a dictionary of colors for each strain in y_strain
    my_color = {'DA': 'red', 'LEW': 'blue'}
    strain_color = [my_color[y_strain[i]] for i in range(len(y_strain))]


    ### make a dictionary of colors for each 16 cluster in y_cluster. use np.unique(y_cluster)
    ### generate 16 colors using the following code:
    my_color = {i: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                        for i in np.unique(y_cluster)}
    cluster_color = [my_color[y_cluster[i]] for i in range(len(y_cluster))]

    return {'sample': sample_color, 'strain': strain_color, 'cluster':cluster_color}



def plot_pca(pca_scores, 
                   num_components_to_plot, 
                   cell_color_vec, 
                   legend_handles=False,
                   covariate=None,
                   plt_legend_dict = None,
                   title='PCA of the data matrix') -> None:
    '''
    plot the PCA components with PC1 on the x-axis and other PCs on the y-axis
    pca_scores: the PCA scores for all the cells
    num_components_to_plot: the number of PCA components to plot as the y-axis
    cell_color_vec: the color vector for each cell
    covariate: the covariate to color the cells
    legend_handles: whether to show the legend handles
    plt_legend_dict: a dictionary of legend handles for each covariate
    covariate: the covariate to color the cells
    title: the title of the plot
    '''
    
    for i in range(1, num_components_to_plot):
        ## color PCA based on strain
        plt.figure()
        ### makke the background white with black axes
        plt.rcParams['axes.facecolor'] = 'white'
        
        plt.scatter(pca_scores[:,0], pca_scores[:,i], c=cell_color_vec, s=1) 
        plt.xlabel('PC1')
        plt.ylabel('PC'+str(i+1))
        plt.title(title)
        if legend_handles:
            plt.legend(handles=plt_legend_dict[covariate])
        plt.show()



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
        ### makke the background white with black axes
        plt.rcParams['axes.facecolor'] = 'white'
        
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
    plt.rcParams['axes.facecolor'] = 'white'
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cell_color_vec, s=1)
    plt.title(title)
    plt.legend(handles=plt_legend_dict[covariate])
    
    plt.show()



def plot_metric_barplot(metric_all, title) -> None:
    '''
    plot the factor metric as a bar plot
    metric_all: the metric value for all the factors
    title: the title of the plot
    '''
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(metric_all)), metric_all)
    ax.set_xticks(range(len(metric_all)))
    x_axis_label = ['F'+str(i) for i in range(1, len(metric_all)+1)]
    ax.set_xticklabels(x_axis_label, rotation=45, ha="right",)
    ax.set_title(title)
    plt.show()



def plot_factor_scatter(factor_scores, x_i, y_i, cell_color_vec, covariate=None,title='') -> None:
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
    if covariate:
        plt.legend(handles=plt_legend_dict[covariate])
    plt.title(title)
    plt.show()



def plot_metric_correlation_clustermap(all_metrics_df) -> None:
      '''
      plot correlation plot with columnns sorted by the dendrogram using seaborn clustermap
      all_metrics_df: a pandas dataframe of the metrics for all the factors
      '''
      ### convert to a numpy array
      all_metrics_np = all_metrics_df.to_numpy()
      factor_metrics = all_metrics_df.columns
      
      ### calculate the correlation matrix
      corr = np.corrcoef(all_metrics_np.T)
      ### calculate the linkage matrix
      
      Z = linkage(corr, 'ward')
      ### plot the correlation matrix with hierarchical clustering
      
      sns.set(font_scale=2.5)
      
      g = sns.clustermap(corr, row_linkage=Z, col_linkage=Z, cmap='viridis', figsize=(19, 13),  #viridis, coolwarm
                         linewidths=.5, linecolor='white') # annot=False, fmt='.4g'
      plt.setp(g.ax_heatmap.get_xticklabels(),rotation=40, ha="right")
      plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
      

      ## add factor metric names to the heatmap column and row labels
      g.ax_heatmap.set_xticklabels(factor_metrics)
      g.ax_heatmap.set_yticklabels(factor_metrics)
      
      plt.show()



def plot_factor_dendogram(factor_loading, distance='ward',num_var=100) -> None:
    '''
    plot a dendogram for the factors based on the correlation matrix distance
    factor_loading: a pandas dataframe of the factor loadings for all the factors
    distance: the distance metric for the linkage matrix
    num_var: the number of genes to use for the dendogram
    '''
    ### convert to a numpy array
    ### factor names are F1, F2, ..., Fn
    
    factor_names = ['F'+str(i+1) for i in range(factor_loading.shape[1])]

    ## select variable genes
    var_genes = np.var(factor_loading, axis=1)
    ## sort the genes based on variance and select the top num_var genes
    var_genes_idx = np.argsort(var_genes)[::-1][0:num_var]

    ### select the top num_var genes for the factor loading matrix numpy array
    factor_loading = factor_loading[var_genes_idx,:]

    ### calculate the correlation matrix
    corr = np.corrcoef(factor_loading.T)
    ### calculate the linkage matrix
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(corr, distance)
    #### make the background white
    
    ## add title including the number of genes used for the dendogram

    plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=factor_names, leaf_rotation=90, leaf_font_size=17)
    plt.rcParams['axes.facecolor'] = 'white'

    plt.rcParams['ytick.labelsize'] = 14
    ## add a box around the dendrogram
    plt.rcParams['axes.linewidth'] = 2
    plt.title('Dendrogram of the factors using the top '+str(num_var)+' genes')
    plt.show()




def plot_metric_dendrogram(all_metrics_df, distance='ward') -> None:
    '''
    plot a dendogram for the metrics based on the correlation matrix distance
    all_metrics_df: a pandas dataframe of the metrics for all the factors
    '''
    ### convert to a numpy array
    all_metrics_np = all_metrics_df.to_numpy()
    factor_metrics = all_metrics_df.columns

    ### calculate the correlation matrix
    corr = np.corrcoef(all_metrics_np.T)
    ### calculate the linkage matrix
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(corr, distance)
    #### make the background white
    plt.rcParams['axes.facecolor'] = 'white'
    ### make the y-axis labels forn size smaller
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    ## add a box around the dendrogram
    plt.rcParams['axes.linewidth'] = 2

    plt.figure(figsize=(5, 5))
    dendrogram(Z, labels=factor_metrics, leaf_rotation=90)
    plt.show()




def plot_metric_heatmap(all_metrics_scaled, factor_metrics, 
                           title='Scaled factor metrics for all factors'):
    '''
    plot the heatmap of the scaled factor metrics for all the factors using seaborn
    all_metrics_scaled: the scaled factor metrics for all the factors
    factor_metrics: the list of factor metric names
    '''
    sns.set(font_scale=1.4)
    plt.figure(figsize=(27,15))
    #all_metrics_np = all_metrics_df.T.to_numpy()
    all_metrics_np = all_metrics_scaled.T
    
    ### remove numbers from heatmap cells

    g = sns.clustermap(all_metrics_np, cmap='YlGnBu', figsize=(28, 20),  #viridis, coolwarm
                            linewidths=.5, linecolor='white',
                            col_cluster=False, row_cluster=True) # annot=False, fmt='.4g'
    plt.setp(g.ax_heatmap.get_xticklabels(),rotation=40, ha="right", fontsize = 30)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize = 30)
    ## add F1, F2, ... to the heatmap x axis

    g.ax_heatmap.set_xticklabels(['F'+str(i+1) for i in range(all_metrics_np.shape[1])])

    g.ax_heatmap.set_yticklabels(factor_metrics)

    plt.show()



def plot_all_factors_levels_df(all_factors_df, title='', color="YlOrBr"):
    '''
    colors: SV: 'RdPu', AUC/: 'YlOrBr', wilcoxon: rocket_r, featureImp: coolwarm
    plot the score of all the factors for all the covariate levels
    all_factors_df: a dataframe of a score for all the factors for all the covariate levels
    '''
    fig, ax = plt.subplots(figsize=(20,5))
    ax = sns.heatmap(all_factors_df, cmap=color, linewidths=.5, annot=False)
    ax.set_title(title)
    ax.set_xlabel('Factors')
    ax.set_ylabel('Covariate level')
    ## add F1, F2, ... to the x-axis ticks
    x_axis_label = ['F'+str(i) for i in range(1, all_factors_df.shape[1]+1)]
    ax.set_xticklabels(x_axis_label, rotation=45, ha="right",)
    plt.show()


def plot_all_factors_levels(SV_all_factors, covariate_levels, title='', color='RdPu'):
    ''' 
    plot the scaled variance of all the factors for each covariate level
    SV_all_factors: the scaled variance of all the factors for each covariate level
    covariate_levels: the list of covariate levels
    '''
    plt.figure(figsize=(16, 3))
    plt.imshow(SV_all_factors, cmap=color, interpolation='nearest')
    plt.yticks(np.arange(len(covariate_levels)), covariate_levels)
    ## add F1, F2, ... to the x-axis ticks
    x_axis_label = ['F'+str(i) for i in range(1, len(SV_all_factors[0])+1)]
    plt.xticks(np.arange(len(SV_all_factors[0])), x_axis_label, rotation=45, ha="right",)
    plt.colorbar()
    plt.xlabel('Factors')
    plt.ylabel('Covariate levels')
    plt.title(title)
    plt.show()



def get_metric_category_color_df() -> (pd.DataFrame, dict):
      '''
      get a dataframe of metric names and their category annd color annotations, plus a dictionary of category and color

      '''
      Homogeneity = ['ASV_protocol_arith', 'ASV_protocol_geo', 'ASV_cell_line_arith', 'ASV_cell_line_geo']
      Effect_Size = ['factor_variance']
      Bimodality = ['bic_km', 'calinski_harabasz_km', 'davies_bouldin_km', 'silhouette_km',
                        'vrs_km', 'wvrs_km', 'bic_gmm', 'silhouette_gmm', 'vrs_gmm', 'wvrs_gmm',
                        'likelihood_ratio', 'bimodality_index', 'dip_score', 'dip_pval', 'kurtosis', 'outlier_sum']
      Complexity = ['factor_entropy']
      Specificity = ['factor_specificity']


      ### make a dataframe of metric names and their category annotations
      metric_category_df = pd.DataFrame({'metric': Homogeneity+Effect_Size+Bimodality+Complexity+Specificity,
                                          'Group': ['Homogeneity']*len(Homogeneity) + ['Effect_Size']*len(Effect_Size) + ['Bimodality']*len(Bimodality) + ['Complexity']*len(Complexity) + ['Specificity']*len(Specificity)})
      

      ### make color dictionary for the categories based on hex values. example: '#1E93AE'
      category_colors_dict = {'Homogeneity': '#1E93AE', 'Effect_Size': '#F9C80E', 
                              'Bimodality': '#F86624', 'Complexity': '#EA3546', 'Specificity': '#662E9B'}

      #category_colors_dict = {'Homogeneity': '#ED2323', 'Effect_Size': '#60FD00', 
      #                        'Bimodality': '#808080', 'Complexity': '#EA3546', 'Specificity': '#662E9B'}

      ### make a row annotation dataframe with metric names and their category annd color annotations
      metric_colors_df = pd.DataFrame({'metric': metric_category_df.metric, 
                              'Group': metric_category_df.Group, 
                              'color': metric_category_df.Group.map(category_colors_dict)})
      
      return metric_colors_df, category_colors_dict





def plot_annotated_metric_heatmap(all_metrics_scaled, factor_metrics):
      '''
        plot the heatmap of the scaled factor metrics for all the factors using seaborn, with row annotations and dendrogram
        all_metrics_scaled: the scaled factor metrics for all the factors
        factor_metrics: the list of factor metric names
        title: the title of the plot
      '''
      
      ### make a dict from metric annd color columns with metric as key and color as value
      metric_colors_df, category_colors_dict = get_metric_category_color_df()
      metric_colors_dict = dict(zip(metric_colors_df.metric, metric_colors_df.color))

      ### convert to dataframe
      all_metrics_scaled_df = pd.DataFrame(all_metrics_scaled, columns=factor_metrics).T
      all_metrics_scaled_df.columns = ['F'+str(i) for i in range(1, all_metrics_scaled.shape[0]+1)]

      factor_metrics = pd.Series(all_metrics_scaled_df.index.values)
      row_colors = factor_metrics.map(metric_colors_dict)
      row_colors.columns = ['Metric type']

      sns.set(font_scale=1.6)
      plt.figure(figsize=(27,15))

      ### remove numbers from heatmap cells
      g = sns.clustermap(all_metrics_scaled_df.reset_index(drop=True), 
                         row_colors=row_colors, cmap='YlGnBu', figsize=(35, 20),  #viridis, coolwarm
                              linewidths=.5, linecolor='white',
                              col_cluster=False, row_cluster=True) # annot=False, fmt='.4g'
      
      plt.setp(g.ax_heatmap.get_xticklabels(),rotation=40, ha="right", fontsize = 30)
      plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize = 30)

      g.ax_heatmap.set_xticklabels(['F'+str(i+1) for i in range(all_metrics_scaled_df.shape[1])])
      g.ax_heatmap.set_yticklabels(factor_metrics)
      #g.fig.suptitle(title, fontsize=30)

      ### make a legennd for the row colors
      for label in category_colors_dict:
            g.ax_row_dendrogram.bar(0, 0, color=category_colors_dict[label],
                                    label=label, linewidth=0)
      g.ax_row_dendrogram.legend(loc="center", ncol=2, bbox_to_anchor=(1.1, 1.1), fontsize=21)
      plt.show()



#### Visualizing global metrics for how well a factor analysis works on a dataset
### visualize the distribution of the matched factors
def plot_matched_factor_dist(matched_factor_dist, title=''):
      plt.figure(figsize=(np.round(len(matched_factor_dist)/3),4))
      plt.bar(np.arange(len(matched_factor_dist)), matched_factor_dist)
      ### add F1, F2, ... to the xticks
      plt.xticks(np.arange(len(matched_factor_dist)), ['F'+str(i) for i in range(1, len(matched_factor_dist)+1)])
      ### make the xticks vertical and set the fontsize to 14
      plt.xticks(rotation=90, fontsize=12)
      #plt.xlabel('Number of matched covariates')
      plt.ylabel('Number of matched covariate levels')
      plt.title(title)
      plt.show()


def plot_matched_covariate_dist(matched_covariate_dist, covariate_levels , title=''):
      plt.figure(figsize=(np.round(len(matched_covariate_dist)/3),4))
      plt.bar(np.arange(len(matched_covariate_dist)), matched_covariate_dist)
      ### add covariate levels to the xticks
      plt.xticks(np.arange(len(matched_covariate_dist)), covariate_levels)

      ### make the xticks vertical and set the fontsize to 14
      plt.xticks(rotation=90, fontsize=12)
      #plt.xlabel('Number of matched factors')
      plt.ylabel('Number of matched factors')
      plt.title(title)
      plt.show()



def plot_histogram(values, xlabel='scores',title='', bins=100,threshold=None) -> None:
      '''
      plot the histogram of a list of values
      values: list of values
      xlabel: xlabel of the histogram
      title: title of the histogram
      '''
      plt.figure(figsize=(10, 5))
      plt.hist(values, bins=bins)
      ## adding a line for a threshold value
      if threshold:
            plt.axvline(x=threshold, color='red')
      plt.xlabel(xlabel)
      plt.ylabel('Frequency')
      plt.title(title)
      plt.show()


