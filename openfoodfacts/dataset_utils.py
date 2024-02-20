"""
Utility functions for loading and preprocessing datasets.
"""

import os
import pandas as pd


def load_dataset(path):
    """
    Load a dataset from a CSV file.

    Parameters:
    - path (str): The file path to the CSV file.

    Returns:
    - df (DataFrame): The loaded dataset as a Pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found in path - {path}")
    print("Loading dataset")
    df = pd.read_csv(path, on_bad_lines="skip", sep="\t")
    # df = pd.read_csv(path, on_bad_lines="skip")
    print("DF: ", df.shape)
    return df


def identify_missing_data(df):
    """
    Identify missing data in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.

    Returns:
    - nan_df (DataFrame): DataFrame containing column names and their respective missing percentages.
    """
    print("\n\nIdentifying missing data")
    percent_missing = df.isnull().sum() * 100 / len(df)
    nan_df = pd.DataFrame({"col_name": df.columns, "percent_missing": percent_missing})
    print(nan_df)
    return nan_df


def filter_columns_by_missing_percentage(df, nan_df, PERC=95):
    """
    Filter columns based on missing percentage threshold.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - nan_df (DataFrame): DataFrame containing column names and their respective missing percentages.
    - PERC (int): The percentage threshold for missing values. Default is 95.

    Returns:
    - df (DataFrame): The filtered DataFrame.
    """
    print("\n\n Filtering columns by missing percentage")
    columns = nan_df[nan_df.percent_missing < PERC].col_name.tolist()
    df = df[columns]
    print("Shape after filtering: ", df.shape)
    return df


def drop_unwanted_columns(df, columns_to_drop):
    """
    Drop unwanted columns from the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame.
    - columns_to_drop (list): List of column names to drop.

    Returns:
    - df (DataFrame): The DataFrame with unwanted columns dropped.
    """
    print("\n\n Dropping unwanted columns")
    return df.drop(columns=columns_to_drop, errors="ignore")
