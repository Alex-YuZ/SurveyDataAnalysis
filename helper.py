import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def viz_freq(df, feature_name, top_k=None, norm=True, asc=False):
    """Summarize and visualize value counts in a categorical feature.

    Args:
        df: pandas dataframe with the dataset of the developers survey
        feature_name ([str]): [A categorical feature in the dateset]
        top_k ([int], optional): [Only show the top k members in the categorical feature]. Defaults to None.
        norm ([bool], optional): [True for using relative frequencies. False for absolute freq..]. Defaults to None.
        asc (bool, optional): [Use ascending or descending order]. Defaults to False.
        
    Returns:
        None
    """
    if top_k is None:
        val_counts = df[feature_name].value_counts(normalize=norm, ascending=asc)
        print(val_counts)
        plt.barh(y=val_counts.index.values, width=val_counts.values);
        
    else:
        val_counts = df[feature_name].value_counts(normalize=norm, ascending=asc)
        top_k_vals = val_counts[-top_k:]
        print(top_k_vals)
        plt.barh(y=top_k_vals.index.values, width=top_k_vals.values);
        
        
def get_description(schema, feature_name):
    """Print out full meaning of a given feature name.
    
    Args:
        schema [pd.DataFrame]: the schema of the developers survey
        feature_name [string]: the name of the column you would like to know about
        
    Returns:
        [str]: [the description of the given feature]
    """
    
    schema = schema.set_index('Column')
    try:
        desc = schema.loc[feature_name, :].values[0]
        
    except KeyError:
        return "WARNING: Check your spelling as no feature called {}".format(feature_name)
        
    return desc


def method_suggested_count(df, plot=True):
    """[summary]

    Args:
        df (pd.DataFrame): dataset of the developers survey
        plot (bool, optional): [Whether or not plot the result]. Defaults to True.
        
    Returns:
        None
    """
    couEduSplit = df[df.CousinEducation.notna()].CousinEducation.str.split('; ', n=-1, expand=True)
    series_lst = []
    for i in range(4):
        series_lst.append(couEduSplit[i].value_counts().sort_index())
    cc = pd.concat(series_lst, axis=1).sum(axis=1)
    cc_relative = (cc/cc.sum()).sort_values()
    
    print(cc_relative.sort_values(ascending=False))
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.barh(cc_relative.index, cc_relative.values, height=.6)

        plt.title("Method of Educating Suggested", fontsize=16)
        plt.ylabel("Methods", fontweight='bold', fontsize=12)
        plt.xlabel('Relative Freq.', fontweight='bold', fontsize=12, labelpad=12);
        
        
def higher_ed_mapping(df, edu_levels):
    """Encode education level in given `edu_levels` list as 1 or 0

    Args:
        df (pd.DataFrame): dataset of the developers survey
        edu_levels ([list]): education level needs to be encoded as 1

    Returns:
        df (pd.DataFrame): A new dataframe with `higher_ed` column added
        prop (float): percentage of respondents with higher edu level
    """
    df['higher_ed'] = df.FormalEducation.isin(edu_levels).astype('int')
    prop = df['higher_ed'].mean()
    return df, prop
    
