import sys
sys.path.append('./Code/')

import os
import pandas as pd
import functions_metrics as fmet
import matplotlib.pyplot as plt

###### reading the importance matrix for each model in each residual type
gini_list_dict = {}
for residual_type in ['pearson', 'response', 'deviance']:
    print(residual_type)

    file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/imp/'
    #file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/base/'
    imp_list = [pd.read_csv(file+f) for f in os.listdir(file) if f.startswith("importance_df")]
    #imp_shuffle_pearson = pd.concat(imp_list, ignore_index=True)
    ### make a dictionary with residual type as key and the corresponding specificity list as value
    
    factor_gini_meanimp_dict = {}

    for i in range(len(imp_list)):
        imp_shuffle_res = imp_list[i]

        imp_shuffle_model = imp_shuffle_res.groupby('model')
        ### print the first 5 rows of the each model in the imp_shuffle_model

        models_list = list(imp_shuffle_model.groups.keys())
        for model in models_list:
            imp_shuffle_a_model = imp_shuffle_model.get_group(model)
            #print(imp_shuffle_a_model.head())

            imp_shuffle_a_model = imp_shuffle_a_model[['factor', 'importance', 'covariate_level']]
            mean_importance_df = imp_shuffle_a_model.pivot(index='covariate_level', columns='factor', values='importance')

            ### calculated for the total importance matrix and append to the list
            factor_gini_meanimp = fmet.get_all_factors_gini(mean_importance_df)
            if model in factor_gini_meanimp_dict:
                factor_gini_meanimp_dict[model].append(factor_gini_meanimp)
            else:
                factor_gini_meanimp_dict[model] = [factor_gini_meanimp]
            
            
        
        ## calculated for each factor in the importance matrix
        #factor_simpson_meanimp = fmet.get_all_factors_simpson(mean_importance_df) 
        #factor_entropy_meanimp = fmet.get_factor_entropy_all(mean_importance_df)  
    
    gini_list_dict[residual_type] = factor_gini_meanimp_dict

### gini_list_dict is a dictionary with residual type as key
## values are dictionaries with model name as key and the corresponding gini list as value
gini_list_dict
gini_list_dict_base = {'pearson': {'AUC': [0.16617428267212767],
  'DecisionTree': [0.8634694436288269],
  'KNeighbors_permute': [0.7299616364057296],
  'LogisticRegression': [0.7331098541883921],
  'RandomForest': [0.5995872216902686],
  'XGB': [0.7138019685603496]},
 'response': {'AUC': [0.16411161438426713],
  'DecisionTree': [0.9040825379876202],
  'KNeighbors_permute': [0.749900069363618],
  'LogisticRegression': [0.7375359102106752],
  'RandomForest': [0.6322951122806095],
  'XGB': [0.7584341332796006]},
 'deviance': {'AUC': [0.16335137161086283],
  'DecisionTree': [0.8903115986861357],
  'KNeighbors_permute': [0.7294318626628646],
  'LogisticRegression': [0.7025766834234716],
  'RandomForest': [0.6176133625207837],
  'XGB': [0.7225796703025885]}}


### make a boxplot for each model in each residual type
for key, values in gini_list_dict.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot each group separately
    for key1, values1 in values.items():
        plt.boxplot(values1, positions=[list(values.keys()).index(key1) + 1], labels=[key1])
        ### add a red for for the base model based on the gini_list_dict_base mean depicted by
        plt.boxplot(gini_list_dict_base[key][key1], positions=[list(values.keys()).index(key1) + 1], labels=[key1], 
                    patch_artist=True, boxprops=dict(facecolor="green", alpha=0.99))
    # Create the boxplot
    plt.title('Boxplot of Gini index for different models in '+key+' residual type')
    plt.xlabel('Models')
    plt.ylabel('Values')
    ## rotate the x ticks
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()


###### reading the importance matrix for each mean and scale type in each residual type
gini_list_dict = {}
for residual_type in ['pearson', 'response', 'deviance']:
    print(residual_type)

    #file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/base/'
    file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/meanimp/'
    imp_list = [pd.read_csv(file+f) for f in os.listdir(file) if f.startswith("meanimp_df")]

    factor_gini_meanimp_dict = {}

    for i in range(len(imp_list)):
        imp_shuffle_res = imp_list[i]
        ### concatenate mean_type and scale_type to make a new column
        imp_shuffle_res['mean_scale'] = imp_shuffle_res['mean_type'] + '_' + imp_shuffle_res['scale_type']


        imp_shuffle_meanscale = imp_shuffle_res.groupby('mean_scale')

        meanscale_list = list(imp_shuffle_meanscale.groups.keys())
        for meanscale in meanscale_list:
            imp_shuffle_a_meanscale = imp_shuffle_meanscale.get_group(meanscale)
            print(imp_shuffle_a_meanscale.head())

            ## only include columnns F1-F30
            imp_shuffle_a_meanscale = imp_shuffle_a_meanscale.iloc[:, 1:31]
            factor_gini_meanimp = fmet.get_all_factors_gini(imp_shuffle_a_meanscale)
            if meanscale in factor_gini_meanimp_dict:
                factor_gini_meanimp_dict[meanscale].append(factor_gini_meanimp)
            else:
                factor_gini_meanimp_dict[meanscale] = [factor_gini_meanimp]

        gini_list_dict[residual_type] = factor_gini_meanimp_dict

gini_list_dict_base = {'pearson': {'arithmatic_minmax': [0.34690793085824884],
  'arithmatic_rank': [0.18927983539094634],
  'arithmatic_standard': [0.722751974157659],
  'geometric_minmax': [0.7348356506964893],
  'geometric_rank': [0.22573728869098225],
  'geometric_standard': [0.970277290098785]},
 'response': {'arithmatic_minmax': [0.3533038250699997],
  'arithmatic_rank': [0.20494524093986455],
  'arithmatic_standard': [0.7229683894999508],
  'geometric_minmax': [0.7993235537103558],
  'geometric_rank': [0.23989593123466804],
  'geometric_standard': [0.9808166459876209]},
 'deviance': {'arithmatic_minmax': [0.3625877011815889],
  'arithmatic_rank': [0.19228560998274272],
  'arithmatic_standard': [0.7410427933633456],
  'geometric_minmax': [0.7790450864214854],
  'geometric_rank': [0.23205907186077762],
  'geometric_standard': [0.9660402393992147]}}

### make a boxplot for each meanscale type in each residual type
for key, values in gini_list_dict.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot each group separately
    for key1, values1 in values.items():
        plt.boxplot(values1, positions=[list(values.keys()).index(key1) + 1], labels=[key1])
    # Create the boxplot
    plt.title('Boxplot of Gini index for different mean_scale types in '+key+' residual type')
    plt.xlabel('mean_scale types')
    plt.ylabel('Values')
    ## rotate the x ticks
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()



### make a boxplot for each model in each residual type
for key, values in gini_list_dict.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    # Plot each group separately
    for key1, values1 in values.items():
        plt.boxplot(values1, positions=[list(values.keys()).index(key1) + 1], labels=[key1])
        ### add a red for for the base model based on the gini_list_dict_base mean depicted by
        plt.boxplot(gini_list_dict_base[key][key1], positions=[list(values.keys()).index(key1) + 1], labels=[key1], 
                    patch_artist=True, boxprops=dict(facecolor="green", alpha=0.99))
    # Create the boxplot
    plt.title('Boxplot of Gini index for different models in '+key+' residual type')
    plt.xlabel('Models')
    plt.ylabel('Values')
    ## rotate the x ticks
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()
