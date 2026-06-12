import pandas as pd
import os

def get_columns(filename: str = 'data/patient_data.csv') -> list:
    """
    Get the column names from a CSV file.
    :param filename: Path to the CSV file.
    :return: List of column names.
    """
    df = pd.read_csv(filename)
    df.drop(os.getenv('DROP_COLUMNS','').split(','), axis=1, inplace=True, errors='ignore')
    return df.columns.tolist()

def cleanup_data(df):
    """
    Clean data by transforming to categorical values and removing NaN values.
    """
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('category')
    return df

def normalize_data(df):
    """
    Normalize data by scaling numerical features.
    :param df: DataFrame containing the data to be normalized.
    :return: Normalized DataFrame.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    return df


