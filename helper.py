import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


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


def pct_by_method(df, title=None, plot=True):
    """percentage of edu-level for suggested and holders

    Args:
        df (pd.DataFrame): dataset of the developers survey
        title (str, optional): plot title. Defaults to None.
        plot (bool, optional): Whether or not plot the result. Defaults to True.

    Returns:
        [type]: [description]
    """
    couEduSplit = df[df.CousinEducation.notna()].CousinEducation.str.split('; ', n=-1, expand=True)
    series_lst = []
    for i in range(4):
        series_lst.append(couEduSplit[i].value_counts().sort_index())
    suggested_agg = pd.concat(series_lst, axis=1).sum(axis=1)
    suggetsed_relative = (suggested_agg/suggested_agg.sum()).sort_values()
    
    suggetsed_relative_vals = suggetsed_relative.sort_values(ascending=False)
    
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.barh(suggetsed_relative.index, suggetsed_relative.values, height=.6)

        plt.title(title, fontsize=16)
        plt.ylabel("Methods", fontweight='bold', fontsize=12)
        plt.xlabel('Relative Freq.', fontweight='bold', fontsize=12, labelpad=12);
        
    return suggetsed_relative_vals
        
        
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
   
   
def ord_cat_convert(df, col, order_list):
    """[summary]

    Args:
        df (pd.DataFrame): dataset of the developers survey
        col (str): categorical feature you want to convert to be ordered
        order_list (list): customized order for the feature to be transformed to
    """
    ordered_type = pd.api.types.CategoricalDtype(ordered=True, categories=order_list)
    df[col] = df[col].astype(ordered_type) 


def job_satisfaction_byGroup(df, by_col):
    """Averaged job satisfaction by a given categorical feature

    Args:
        df (pd.DataFrame): dataset of the developers survey
        by_col (str): categorical feature you want to group by
    """
    res_srs = df.groupby(by_col).JobSatisfaction.mean().sort_values(ascending=False)
    res_df = res_srs.rename('Avg_JobSatisfaction').to_frame()
    print(res_df)
    

def cat_na_filter(df, bound=0, print_count=True, print_cols=False):
    """print out categorical variables with certain amount of missing values.

    Args:
        df (pd.DataFrame): dataset of the developers survey
        bound (int or float, optional): pct of missingness. Defaults to 0.
        print_count (bool, optional): print out the count number only. Defaults to True.
        print_cols (bool, optional): print out the variable names. Defaults to False.

    Returns:
        None
    """
    cat_vars = df.select_dtypes('object').columns
    
    if bound < 0:
        return "ERROR: bound must be non-negative!"
    
    if bound==0:
        if print_count:
            count = len(df[cat_vars].columns[df[cat_vars].isna().mean()==0])
            print("There are {} categorical variables with no misisng values.".format(count))

        if print_cols:
            cols_name = df[cat_vars].columns[df[cat_vars].isna().mean()==0]
            pprint(cols_name.to_list())
            
    else:
        if print_count:
            count = len(df[cat_vars].columns[df[cat_vars].isna().mean()>=bound])
            print("There are {0} categorical variables with with more than {1:.2%} of missing values.".format(count, bound))

        if print_cols:
            cols_name = df[cat_vars].columns[df[cat_vars].isna().mean()>=bound]
            pprint(cols_name.to_list())
    