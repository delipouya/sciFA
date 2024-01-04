import os
import pandas as pd
import functions_metrics as fmet
import matplotlib.pyplot as plt


gini_list_dict = {}
for residual_type in ['pearson', 'response', 'deviance']:
    print(residual_type)

    file = '/home/delaram/sciFA/Results/benchmark/'+residual_type+'/shuffle/imp/'
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


### make a boxplot for each residual type
fig, ax = plt.subplots(figsize=(10, 5))
# Plot each group separately
for key, values in gini_list_dict.items():
    plt.boxplot(values, positions=[list(gini_list_dict.keys()).index(key) + 1], labels=[key])
# Create the boxplot
plt.title('Boxplot of Gini index for different residual types')
plt.xlabel('rsidual types')
plt.ylabel('Values')

# Show the plot
plt.show()