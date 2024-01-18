import sys
sys.path.append('./Code/')
import os
import pandas as pd
import functions_metrics as fmet
import matplotlib.pyplot as plt


#########################################################################
###################   Single Model Importance Evaluation  ##################
#########################################################################

###### reading the importance matrix for each model in each residual type
gini_list_dict = {}
for residual_type in ['pearson', 'response', 'deviance']: #, 
    print(residual_type)

    ##### human liver benchmark
    #file = '/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/base/'
    file = '/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/imp/'
    
    #### scMixology benchmark
    #file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/imp_v1/'
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
            mean_importance_df = imp_shuffle_a_model.pivot(index='covariate_level', columns='factor', 
                                                           values='importance')

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

#######################################################
##### scMixology base model
#######################################################
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

gini_list_dict_base ={'pearson': {'AUC': [0.39156280595818843, 0.39156280595818843],
  'DecisionTree': [0.9374897482243939, 0.9365028310136162],
  'KNeighbors_permute': [0.7971112903948738],
  'LogisticRegression': [0.6039708539658788, 0.6039708539658788],
  'RandomForest': [0.6662242305163689],
  'XGB': [0.8182990742252915, 0.8182990742252915]}}
#######################################################


#######################################################
##### Human liver base model
#######################################################

gini_list_dict_base = {'pearson': {'AUC': [0.42844405666813934],
  'DecisionTree': [0.8270269806718208],
  'LogisticRegression': [0.6244436026479022],
  'XGB': [0.6074797580935721]},
 'response': {'AUC': [0.44300881770580863],
  'DecisionTree': [0.8405454456704068],
  'LogisticRegression': [0.7181351178064663],
  'XGB': [0.6105563677029542]},
 'deviance': {'AUC': [0.41502764561149913],
  'DecisionTree': [0.8705934843417716],
  'LogisticRegression': [0.533822322964555],
  'XGB': [0.6682235875773687]}}
#######################################################


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


############## Save results in a dataframe
### write a for loop same as above to make a dataframe for each residual type
for key, values in gini_list_dict.items():
    gini_df = pd.DataFrame.from_dict(values, orient='columns')
    ### add a column for model name
    gini_df['model'] = 'single'
    ### add a column for residual type
    gini_df['residual_type'] = key
    print(gini_df.head())
    gini_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/gini_analysis/'+'imp_gini_df_'+key+'.csv')



#########################################################################
###################    Mean Importance Evaluation  ##################
#########################################################################
###### reading the importance matrix for each mean and scale type in each residual type
gini_list_dict = {}
for residual_type in ['pearson', 'response', 'deviance']: #, 
    print(residual_type)
    

    ##### human liver benchmark
    #file = '/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/base/'
    file = '/home/delaram/sciFA/Results/benchmark_humanliver/'+residual_type+'/shuffle/meanimp/'

    #### scMixology benchmark
    #file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/base/'
    #file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/meanimp/'

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


#############################################
##### scMixology base model
#############################################
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

gini_list_dict_base ={'pearson': {'arithmatic_minmax': [0.4370696143830938],
  'arithmatic_rank': [0.21206342094782954],
  'arithmatic_standard': [0.3395523214640297],
  'geometric_minmax': [0.8769434083699224],
  'geometric_rank': [0.24695374201171472],
  'geometric_standard': [0.3679443575769781]}}

#######################################################


#######################################################
##### Human liver base model
#######################################################
gini_list_dict_base ={'pearson': {'arithmatic_minmax': [0.43145743916258367],
  'arithmatic_rank': [0.21630397132616488],
  'arithmatic_standard': [0.3213467464296892],
  'geometric_minmax': [0.6900014097599303],
  'geometric_rank': [0.2652688750895878],
  'geometric_standard': [0.3446566538678779]},
 'response': {'arithmatic_minmax': [0.4482304042565286],
  'arithmatic_rank': [0.21807670250896058],
  'arithmatic_standard': [0.3348031566839708],
  'geometric_minmax': [0.7105488962352995],
  'geometric_rank': [0.26676924492481086],
  'geometric_standard': [0.3569829820137092]},
 'deviance': {'arithmatic_minmax': [0.4312870752553559],
  'arithmatic_rank': [0.2163803870967742],
  'arithmatic_standard': [0.33342872790755695],
  'geometric_minmax': [0.7297170676765146],
  'geometric_rank': [0.26276100746128445],
  'geometric_standard': [0.3607843057745439]}}

#######################################################

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


### write a for loop same as above to make a dataframe for each residual type
for key, values in gini_list_dict.items():
    gini_df = pd.DataFrame.from_dict(values, orient='columns')
    ### add a column for model name
    gini_df['model'] = 'ensemble'
    ### add a column for residual type
    gini_df['residual_type'] = key
    print(gini_df.head())
    gini_df.to_csv('/home/delaram/sciFA/Results/benchmark_humanliver/gini_analysis/'+'meanimp_gini_df_'+key+'.csv')


