import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

sns.set_style('darkgrid')

def extract_unique_parts(s, pattern):
    # Search for the pattern in the string
    match = re.search(pattern, s)

    # Return the matched groups if found
    if match:
        return match.groups()
    else:
        return None

# Set the directory path
def load_data(foldername):
    flist = os.listdir(foldername)
    
    # Initialize a list to store the data
    ret = dict()
    
    # Iterate over each file in the directory
    for filename in flist:
        if filename.endswith('.json'):
            fullname = os.path.join(foldername, filename)
            with open(fullname, 'r') as file:
                # Load the JSON data from the file and append it to the list
                data = json.load(file)
                ret[fullname] = data
    return ret

def get_df_csl(data, pattern):
    summary = []
    for key, val in data.items():
        match = extract_unique_parts(key, pattern)
        if match:
            seed = int(key.split('_')[-2])
            criterion = key.split('_')[-3]
            setting = val['stage'].split('_')
            epoch, iter = int(setting[1]), int(setting[2])
            soft = 'Soft' if (val['type'] == 'soft') else 'Hard'
            number = val['number']
            student = val['student']
            teacher = val['teacher']  
            rate = val['rate']
            precision = val['precision']
            recall = val['recall']      
            for i in range(len(number)):
                summary.append([epoch, iter, soft, seed, int(number[i]), criterion,
                                float(student[i]), float(teacher[i]), 
                                float(rate[i]), float(precision[i]), float(recall[i])])
    df = pd.DataFrame(data = np.array(summary), columns= ["Epoch", "Iter", "Type", "Seed", "Number", "Criterion",
                                                          "Student", "Teacher", "Rate", "Precision", "Recall"])
    df['Epoch'] = df['Epoch'].astype(int)
    df['Iter'] = df['Iter'].astype(int)
    df['Seed'] = df['Seed'].astype(int)
    df['Number'] = df['Number'].astype(int)
    df['Student'] = df['Student'].astype(float)
    df['Teacher'] = df['Teacher'].astype(float)
    df['Rate'] = df['Rate'].astype(float)
    df['Precision'] = df['Precision'].astype(float)
    df['Recall'] = df['Recall'].astype(float)
    df = df.sort_values(by=['Epoch', 'Iter', 'Type', 'Number', "Seed", "Criterion"], ascending=[True, True, True, True, True, True])
    df.reset_index(drop=True, inplace=True)
    return df

def main(df_epoch, df_iter, df_denoise):
    # load result
    data_all = load_data('result')
    pattern = r'_1_2_4_8_(.*?)_csl.json'
    df = get_df_csl(data_all, pattern)

    # compute Recovery
    ceil = 0.74
    df['Base'] = df['Rate'] * 0

    for epoch in df['Epoch'].unique():
        for iter in df['Iter'].unique():
            for type in df['Type'].unique():
                for seed in df['Seed'].unique():
                    cond = (df['Epoch'] == epoch) & (df['Iter'] == iter) & (df['Type'] == type) & (df['Seed'] == seed)
                    if cond.any():
                        df.loc[cond, 'Base'] = df.loc[cond & (df['Number'] == 1), 'Teacher'].iloc[0]
    df['Recovery'] = (df['Student'] - df['Base']) / (ceil - df['Base']) * 100

    # Dataframe condition
    dft = df[(df['Epoch'] == df_epoch) & (df['Iter'] == df_iter) & (df['Criterion'] == df_denoise)]

    # Calculate the mean and standard deviation for each group
    grouped = dft.groupby(['Number', 'Type'])['Recovery']
    means = grouped.mean().reset_index(name='Recovery_mean')
    stds = grouped.std().reset_index(name='Recovery_std')

    # Merge the standard deviations with the means
    merged_df = pd.merge(means, stds, on=['Number', 'Type'])

    plt.figure(figsize=(5, 4))
    # Use the combined column for hue
    bar_plot = sns.barplot(data=merged_df, x='Number', y='Recovery_mean', hue='Type', errorbar=None)

    # Iterate over the bars to add error bars
    for i, bar in enumerate(bar_plot.patches):
        # Calculate index for merged_df to get the std deviation
        # Considering multiple bars for each 'Number', adjust index calculation if necessary
        data_index = i % len(merged_df['Type'].unique())
        std = merged_df.iloc[data_index]['Recovery_std']  # Adjust this line if necessary
        
        # Position of the error bar
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        
        # Add error bar
        plt.errorbar(x, y, yerr=std, fmt='none', color='black', capsize=5)

    # Define new legend labels (ensure the order and number of labels match your plot)
    new_labels = ['Hard', 'Soft']
    # Set the legend on the plot to use the new labels
    plt.legend(title=None, labels=new_labels, bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2)  # Adjust ncol as needed

    vmin = dft['Recovery'].min() * 0.95
    vmax = dft['Recovery'].max() * 1.05

    plt.xlabel('Number of Weak Supervisors')
    plt.ylabel('PGR (%)')
    plt.ylim([vmin, vmax])  # Adjust ylim according to your data
    plt.tight_layout()

    # Get the current figure using plt.gcf() after plotting
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    figname = 'figure/csl_imagenet.png'  # Updated figname for clarity

    # Save the figure
    fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    print(f'Save figure to {figname}')

if __name__ == "__main__":
    iteration = 1000
    epoch = 0
    denoise = 'top3'
    main(epoch, iteration, denoise)