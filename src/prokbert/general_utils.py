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
    """Checks if a DataFrame contains the expected columns.

    :param df: The input DataFrame to be checked.
    :type df: pd.DataFrame
    :param expected_columns: A list of columns that are expected to be present in the DataFrame.
    :type expected_columns: list
    :param df: pd.DataFrame: 
    :param expected_columns: list: 
    :returns: True if all expected columns are present in the DataFrame, False otherwise.
    :rtype: bool
    :raises ValueError: If any of the expected columns are not present in the DataFrame.

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
    """Checks if a specified column in a DataFrame can serve as a valid primary key.

    :param df: The input DataFrame to be checked.
    :type df: pd.DataFrame
    :param column_name: The name of the column to check.
    :type column_name: str
    :returns: True if the column can serve as a valid primary key, False otherwise.
    :rtype: bool
    :raises ValueError: If the specified column does not exist in the DataFrame.

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
    """Generator that yields non-empty files from a specified directory and its subdirectories based on the given extensions.

    :param start_path: The path to the directory from which to start the search.
    :type start_path: str
    :param extensions: A tuple of file extensions to look for (default is ('.fasta', '.fna')).
                       The function also automatically checks for compressed versions with '.gz'.
    :type extensions: tuple
    :returns: Yields filenames that match the specified extensions and are non-empty.
    :rtype: str

    """
    
    for dirpath, _, filenames in os.walk(start_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if any(filename.endswith(ext) or filename.endswith(ext + '.gz') for ext in extensions) and os.path.getsize(filepath) > 0:
                yield filename



def truncate_zero_columns(arr: np.ndarray) -> np.ndarray:
    """Truncate all trailing columns composed entirely of zeros in a given 2D numpy array.

    :param arr: Input 2D numpy array.
    :type arr: np.ndarray
    :returns: A new array with trailing zero columns removed.
    :rtype: np.ndarray

    """
    
    # Iterate over columns from the end
    for idx in range(arr.shape[1]-1, -1, -1):
        if np.any(arr[:, idx]):
            return arr[:, :(idx+1)]
    return np.empty((arr.shape[0], 0))


import os

def create_directory_for_filepath(filepath: str) -> None:
    """Given a file path, creates the underlying directory structure if it doesn't already exist.

    :param filepath: The path to the file for which the directory structure should be created.
    :type filepath: str 
    :raises ValueError: If the provided path is empty or None.
    :raises OSError: If there's an error creating the directory structure.

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
    """Checks if the provided file path exists.

    :param file_path: Path to the file.
    :type file_path: str 
    :returns: True if the file exists, raises ValueError otherwise.
    :rtype: bool

    """
    if os.path.exists(file_path):
        return True
    else:
        raise ValueError(f"The provided file path '{file_path}' does not exist.")

def count_gpus(method="clinfo"):
    """
    Count the number of available GPUs using the specified method.

    This function counts the number of NVIDIA and AMD GPUs using the chosen method. By default, it uses the 'clinfo'
    method for AMD GPUs.

    :param method: The method to use for GPU counting. Choose between 'clinfo' (default) and 'rocm'.
    :type method: str, optional

    :return: The total number of GPUs detected.
    :rtype: int

    :raises ValueError: If an unknown method is provided.

    :raises Exception: If an error occurs while querying AMD GPUs using the specified method.

    .. note::
        - The 'clinfo' method queries AMD GPUs by running the 'clinfo' command.
        - The 'rocm' method queries AMD GPUs by running 'rocm-smi --list' command.

    """
    import torch
    import subprocess

    # Count NVIDIA GPUs
    nvidia_gpu_count = torch.cuda.device_count()

    # Count AMD GPUs
    amd_gpu_count = 0
    try:
        if method == "clinfo":
            clinfo_output = subprocess.check_output('clinfo').decode('utf-8')
            amd_gpu_count = clinfo_output.lower().count('device type: gpu')
        elif method == "rocm":
            rocm_output = subprocess.check_output('rocm-smi --list', shell=True).decode('utf-8')
            amd_gpu_count = len(rocm_output.strip().split('\n'))
        else:
            raise ValueError("Unknown method provided. Choose between 'clinfo' and 'rocm'.")
    except Exception as e:
        print(f"Error querying AMD GPUs using method '{method}': {e}")

    total_gpus = nvidia_gpu_count + amd_gpu_count

    return total_gpus



def create_hard_links(source_directory: str, target_directory: str, blacklist: list = []) -> None:
    """Creates hard links for all files from the source directory to the target directory.

    :param source_directory: The directory containing the original files.
    :type source_directory: str
    :param target_directory: The directory where hard links will be created.
    :type target_directory: str
    :param blacklist: List of filenames to exclude from creating hard links.
    :type blacklist: list
    :returns: None

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
    """Creates hard links for the specified files from the source directory to the target directory.

    :param source_directory: The directory containing the original files.
    :type source_directory: str
    :param target_directory: The directory where hard links will be created.
    :type target_directory: str
    :param filenames: List of filenames for which hard links should be created.
    :type filenames: list
    :returns: None

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
    """Removes all files recursively in a folder that start with '.' or '_'.

    :param directory: The directory from which hidden files should be removed.
    :type directory: str
    :returns: None

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
    