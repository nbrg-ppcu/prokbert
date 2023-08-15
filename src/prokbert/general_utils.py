# coding=utf-8

import pandas as pd
import os
import numpy as np
import subprocess
import shutil
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


import os

def create_directory_for_filepath(filepath: str) -> None:
    """
    Given a file path, creates the underlying directory structure if it doesn't already exist.

    Args:
        filepath (str): The path to the file for which the directory structure should be created.

    Raises:
        ValueError: If the provided path is empty or None.
        OSError: If there's an error creating the directory structure.
    """
    
    if not filepath:
        raise ValueError("The provided filepath is empty or None.")
    
    directory = os.path.dirname(filepath)
    
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory structure {directory} created successfully.")
        except OSError as e:
            raise OSError(f"Error creating directory structure {directory}. Error: {e}")

# Example usage:
# create_directory_for_filepath("/path/to/directory/that/might/not/exist/filename.txt")

def check_file_exists(file_path: str) -> bool:
    """
    Checks if the provided file path exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, raises ValueError otherwise.
    """
    if os.path.exists(file_path):
        return True
    else:
        raise ValueError(f"The provided file path '{file_path}' does not exist.")

def count_gpus():
    # Count NVIDIA GPUs
    import torch
    nvidia_gpu_count = torch.cuda.device_count()

    # Count AMD GPUs
    amd_gpu_count = 0
    try:
        clinfo_output = subprocess.check_output('clinfo').decode('utf-8')
        amd_gpu_count = clinfo_output.count('Device Type: GPU')
    except:
        pass  # clinfo command might not be available

    total_gpus = nvidia_gpu_count + amd_gpu_count

    return total_gpus


def create_hard_links(source_directory: str, target_directory: str, blacklist: list = []) -> None:
    """
    Creates hard links for all files from the source directory to the target directory.
    
    Args:
        source_directory (str): The directory containing the original files.
        target_directory (str): The directory where hard links will be created.
        blacklist (list): List of filenames to exclude from creating hard links.
    
    Returns:
        None
    """
    
    # Ensure the provided directories exist
    if not os.path.exists(source_directory):
        raise ValueError(f"The source directory '{source_directory}' does not exist.")
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Iterate through the files in the source directory
    for filename in os.listdir(source_directory):
        source_file_path = os.path.join(source_directory, filename)
        target_file_path = os.path.join(target_directory, filename)
        
        # Check for files to skip
        if (filename.startswith('.') or 
            filename.startswith('_') or 
            os.path.isdir(source_file_path) or 
            filename in blacklist):
            continue
        
        # Create a hard link
        os.link(source_file_path, target_file_path)

    return f"Hard links created in {target_directory} from {source_directory}."

# Example usage
# create_hard_links("/path/to/source_directory", "/path/to/target_directory", blacklist=["file_to_skip.txt"])

def create_selected_hard_links(source_directory: str, target_directory: str, filenames: list) -> None:
    """
    Creates hard links for the specified files from the source directory to the target directory.
    
    Args:
        source_directory (str): The directory containing the original files.
        target_directory (str): The directory where hard links will be created.
        filenames (list): List of filenames for which hard links should be created.
    
    Returns:
        None
    """
    
    # Ensure the provided directories exist
    if not os.path.exists(source_directory):
        raise ValueError(f"The source directory '{source_directory}' does not exist.")
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    # Iterate through the specified filenames
    for filename in filenames:
        source_file_path = os.path.join(source_directory, filename)
        target_file_path = os.path.join(target_directory, filename)
        
        # Ensure the file exists in the source directory
        if not os.path.isfile(source_file_path):
            print(f"Warning: {filename} does not exist in the source directory. Skipping.")
            continue
        
        # Create a hard link
        try:
            os.link(source_file_path, target_file_path)
        except FileExistsError:
            print(f'The target hard link {target_file_path} exist. Skipping...')

    return f"Hard links for specified files created in {target_directory} from {source_directory}."

def remove_hidden_files(directory: str) -> None:
    """
    Removes all files recursively in a folder that start with '.' or '_'.
    
    Args:
        directory (str): The directory from which hidden files should be removed.
    
    Returns:
        None
    """
    
    # Ensure the directory exists
    if not os.path.exists(directory):
        raise ValueError(f"The directory '{directory}' does not exist.")
    
    # Use os.walk to iterate through all subdirectories and files
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        
        # Filter out directories starting with '.' or '_'
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and not d.startswith('_')]
        
        # Remove files starting with '.' or '_'
        for filename in filenames:
            if filename.startswith('.') or filename.startswith('_'):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Removed: {file_path}")
                
    print(f"All hidden files removed from {directory}.")
    