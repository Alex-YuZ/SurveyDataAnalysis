import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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
    """Convert a categorical (nominal) variable into ordinal one.

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
            
def create_dummy_df(df, cat_cols, dummy_na=True):
    '''Create dummied dataframe by one-hot encoding.
    
    INPUT:
    df - raw pandas dataframe with all variables
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    '''
    for col in  cat_cols:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(col, axis=1), pd.get_dummies(df[col], 
                                                             prefix=col, 
                                                             prefix_sep='_', 
                                                             drop_first=True, 
                                                             dummy_na=dummy_na)], axis=1)
    return df

def clean_fit_linear_mod(df, response_col, dummy_na, test_size=.3, random_state=42):
    """Use imputed numeric variables and one-hot-encoded catgorical variables for prediction

    Args:
        df (pd.DataFrame): dataset of the developers survey.
        response_col (str): target variable name you want to predict on.
        dummy_na (bool): whether to encode np.nan as dummies.
        test_size (float, optional): percentage of testing data. Defaults to .3.
        random_state (int, optional): an int that is provided as the random state for 
          splitting the data into training and test. Defaults to 42.
          

    Returns:
        train_score - float - r2 score on the train data
        test_score - float - r2 score on the test data
        lm_model - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    """
    
    # Drop the rows with missing response values
    df = df.dropna(subset=[response_col])
    
    # Drop columns with NaN for all the values
    df = df.dropna(how='all', axis=1)
    
    # Use create_dummy_df to dummy categorical columns
    cat_cols = df.select_dtypes('object').columns
    df = create_dummy_df(df, cat_cols, dummy_na=dummy_na)
    
    # Fill the mean of the column for any missing values
    fill_mean = lambda col: col.fillna(col.mean())
    df = df.apply(fill_mean)
    
    # Split your data into an X matrix and a response vector y
    X = df.drop(columns=[response_col])
    y = df[response_col]
    
    # Create training and test sets of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Instantiate a LinearRegression model with normalized data
    lm = LinearRegression(normalize=True)
    
    # Fit the model to the training data
    lm.fit(X_train, y_train)
    
    # Predict response and calculate metric on the training data
    y_train_preds = lm.predict(X_train)
    train_score = r2_score(y_train, y_train_preds)
    
    # Predict response and calculate metric on the testing data
    y_test_preds = lm.predict(X_test)
    test_score = r2_score(y_test, y_test_preds)
    
    return train_score, test_score, lm, X_train, X_test, y_train, y_test
    