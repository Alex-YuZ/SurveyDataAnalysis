import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def viz_freq(df, feature_name, top_k=None, norm=True, asc=False):
    """Summarize and visualize value counts in a categorical feature.

    Args:
        df: dataframe
        feature_name ([str]): [A categorical feature in the dateset]
        top_k ([int], optional): [Only show the top k members in the categorical feature]. Defaults to None.
        norm ([bool], optional): [True for using relative frequencies. False for absolute freq..]. Defaults to None.
        asc (bool, optional): [Use ascending or descending order]. Defaults to False.
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