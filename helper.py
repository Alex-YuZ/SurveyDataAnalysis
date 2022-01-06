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