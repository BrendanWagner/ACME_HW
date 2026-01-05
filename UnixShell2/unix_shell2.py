# shell2.py
"""Volume 3: Unix Shell 2.
<Name>
<Class>
<Date>
"""

import os
import subprocess
from glob import glob
from binaryornot.check import is_binary

# Problem 3 Helper Function
def is_valid_file(filename: str) -> bool:
    return not is_binary(filename)

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
                match the file_pattern.
        file_pattern (str): Specifies which files to search.

    Returns:
        matched_files (list): list of the filenames that matched the file
               pattern AND the target string.
    """
    the_ones_that_work = []
    files = glob(f'**/{file_pattern}', recursive=True) # From the workbook
    for file in files:
        if not is_valid_file(file): # If it's fine
            continue
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            if target_string in f.read(): # If it matches the string
                the_ones_that_work.append(file) # Add it
    return the_ones_that_work



# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    my_dict = {}    
    for idk in os.walk("Shell2"):
        print(idk)

    return my_dict

# target_string = "range"
# file_pattern = "**/*.py"

# print(largest_files(10))