# coding=utf-8

import pandas as pd
import os
import numpy as np
""" Library for general utils, such as dataframe properties checking,
creating directories, checking files, etc.
"""


def check_expected_columns(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Checks if a DataFrame contains the expected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be checked.
    expected_columns : list
        A list of columns that are expected to be present in the DataFrame.
        
    Returns
    -------
    bool
        True if all expected columns are present in the DataFrame, False otherwise.
        
    Raises
    ------
    ValueError
        If any of the expected columns are not present in the DataFrame.
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> check_expected_columns(df, ['A', 'B'])
    True
    
    >>> check_expected_columns(df, ['A', 'C'])
    ValueError: The following columns are missing: ['C']
    """
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"The following columns are missing: {missing_columns}")
    
    return True


def is_valid_primary_key(df: pd.DataFrame, column_name: str) -> bool:
    """
    Checks if a specified column in a DataFrame can serve as a valid primary key.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be checked.
    column_name : str
        The name of the column to check.
        
    Returns
    -------
    bool
        True if the column can serve as a valid primary key, False otherwise.
        
    Raises
    ------
    ValueError
        If the specified column does not exist in the DataFrame.
        
    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> is_valid_primary_key(df, 'A')
    True
    
    >>> df = pd.DataFrame({'A': [1, 2, 2], 'B': [4, 5, 6]})
    >>> is_valid_primary_key(df, 'A')
    False
    """
    
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Check for NaN values
    if df[column_name].isnull().any():
        return False
    
    # Check for unique values
    if not df[column_name].is_unique:
        return False
    
    return True

def get_non_empty_files(start_path: str, extensions: tuple = ('.fasta', '.fna')) -> str:
    """
    Generator that yields non-empty files from a specified directory and its subdirectories based on the given extensions.

    :param start_path: The path to the directory from which to start the search.
    :type start_path: str

    :param extensions: A tuple of file extensions to look for (default is ('.fasta', '.fna')).
                       The function also automatically checks for compressed versions with '.gz'.
    :type extensions: tuple

    :return: Yields filenames that match the specified extensions and are non-empty.
    :rtype: str
    """
    
    for dirpath, _, filenames in os.walk(start_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if any(filename.endswith(ext) or filename.endswith(ext + '.gz') for ext in extensions) and os.path.getsize(filepath) > 0:
                yield filename



def truncate_zero_columns(arr: np.ndarray) -> np.ndarray:
    """
    Truncate all trailing columns composed entirely of zeros in a given 2D numpy array.
    
    :param arr: Input 2D numpy array.
    :type arr: np.ndarray

    :return: A new array with trailing zero columns removed.
    :rtype: np.ndarray
    """
    
    # Iterate over columns from the end
    for idx in range(arr.shape[1]-1, -1, -1):
        if np.any(arr[:, idx]):
            return arr[:, :(idx+1)]
    return np.empty((arr.shape[0], 0))
