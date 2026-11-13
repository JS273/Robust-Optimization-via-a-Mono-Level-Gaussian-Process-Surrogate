"""File-management, logging, and experiment reproducibility utilities.

This module provides utilities for creating and managing result directories,
removing outdated experiment folders, copying executed scripts and files,
initializing loggers, retrieving Git commit information, and locating the
latest temporary result folder.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import re
from pathlib import Path
from datetime import datetime
import os, logging, shutil
from datetime import datetime, date
import subprocess

def all_subdirs_of(b='.'):
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(d)
    return result

def create_result_folder(tag = "", max_res_folders = None, max_daily_folders = None, archive = False, hpc = False):

    tag = tag.replace(" ", "_")

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%Y-%m-%d")

    # Folder path
    if archive:
        path = "_temp/Archive/" + d1
    elif hpc:
        path = "_temp/Archive/HPC_Samples/" + d1
    else:
        path = "_temp/" + d1

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    # Keep only last 7 daily result folders
    if not archive:
        if max_daily_folders is not None:
            date_format = '%Y-%m-%d'
            remove_folders("./_temp", max_daily_folders, date_format)

    # Output Folder
    now = datetime.now()
    time_string = now.strftime("%H_%M_%S")
    target_folder_path = path + '/' + time_string + "_" + tag 

    # Remove Output folders greater than 
    if not archive:
        if max_res_folders is not None:
            date_format = '%H_%M_%S'
            remove_folders(path, max_res_folders, date_format)

    # Create target folder
    os.makedirs(target_folder_path)

    return target_folder_path

def remove_folders(path, max_keep, date_format):
    
        folders = all_subdirs_of(path)
        date_obj = []
        if date_format == '%Y-%m-%d':
            max_idx = 10
        elif date_format == '%H_%M_%S':
            max_idx = 8

        for day_folder in folders:
            
            try:
                date_obj.append(datetime.strptime(day_folder[0:max_idx], date_format))
            except ValueError:
                if day_folder != "Archive":
                    print(f'Could not interprete foldername: "{day_folder}" . Skiped folder')

        remove_staps = []

        if len(date_obj) > max_keep:
            n_remove = len(date_obj) - max_keep
            date_obj.sort()

            for i in range(n_remove):
                remove_staps.append(date_obj[i].strftime(date_format))

            for day_folder in folders:
                for remove_stap in remove_staps:
                    if day_folder[0:max_idx] == remove_stap:
                        shutil.rmtree(path + "/" + day_folder)

def save_script(filepath, tag = "", max_daily_folders = None, max_res_folders = None, archive = False, hpc = False):
    target_folder_path = create_result_folder(tag, max_res_folders, max_daily_folders, archive, hpc)

    # Create target file
    target_file = target_folder_path + "/executed_file_" + tag + ".py"

    # Copy script
    shutil.copy(filepath, target_file)

    return target_folder_path

def get_files_with_extension(folder_path, extension):
    file_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(extension):
            file_list.append(filename)
    return file_list

def init_logger(save_dir, logfile, log_level=logging.INFO):

    log_file = save_dir + "/" + logfile + ".log"

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    commit_hash, branch = get_git_hash()
    pyelpaso_git_hash, pyelpaso_branch = get_package_git_hash("pyelpaso")

    logger.info(f"-------------------------------")
    logger.info(f"--------Start logging ---------")
    logger.info(f"-------------------------------\n")

    if commit_hash is not None:
        logger.info(f"Acoustic-Data-Generator git on branch: {branch} at hash: {commit_hash}")

    if pyelpaso_git_hash is not None:
        logger.info(f"Pyelpaso git                on branch: {pyelpaso_branch} at hash: {pyelpaso_git_hash}")

    return logger

def save_file(result_folder, filepath, create_log = True):
    
    filename_with_extension = os.path.basename(filepath)

    # Create target file
    filename = "/executed_file_" + filename_with_extension
    target_file = result_folder + filename

    if create_log:
        logger = init_logger(result_folder, "executing")
        logger.info(f"Executed notebook: {filepath}")
    
    # Copy script
    shutil.copy(filepath, target_file)

    return filename

def get_git_hash():
    try:
        # Read the contents of the HEAD file
        with open('.git/HEAD', 'r') as head_file:
            ref = head_file.readline().strip()

        # If the HEAD file contains a reference to a branch, extract the branch name
        if ref.startswith('ref:'):
            branch_ref = ref.split(':')[-1].strip()
            branch_name = os.path.basename(branch_ref)
            
            # Read the contents of the branch reference file to get the commit hash
            with open(os.path.join('.git', branch_ref), 'r') as branch_ref_file:
                commit_hash = branch_ref_file.readline().strip()[:8]
        else:
            # Otherwise, the HEAD file contains the commit hash directly
            commit_hash = ref
            branch_name = "DETACHED"

        return commit_hash, branch_name
    except FileNotFoundError:
        return None, None

def get_package_git_hash(package, levels_up = 2):

    try:
        # Get the path to the package's __init__.py file
        package_init_file = __import__(package).__file__
        package_dir = os.path.dirname(package_init_file)

        for i in range(levels_up):
            package_dir = os.path.dirname(package_dir)

        # Navigate to the package's directory
        current_directory = os.getcwd()
        os.chdir(package_dir)

        # Execute Git command to get the hash of the current commit
        git_hash, branch_name = get_git_hash()

        os.chdir(current_directory)
        return git_hash, branch_name
    except:
        return None, None
    
def print_git_hash():
    commit_hash, branch = get_git_hash()
    print(f"Git on branch: {branch} at hash: {commit_hash}")


def get_latest_temp_subfolder():
    """
    Returns the name of the latest subfolder within the parent directory based on the hh_mm_ss timestamp prefix.

    Parameters:
    - parent_dir (str or Path): The path to the parent directory.

    Returns:
    - str: The name of the latest subfolder.
    - None: If no matching subfolders are found.
    """

    parent_dir = f"_temp/{date.today()}"

    # Ensure parent_dir is a Path object
    parent_path = Path(parent_dir)
    
    # Verify that the parent directory exists
    if not parent_path.is_dir():
        raise ValueError(f"The provided path '{parent_dir}' is not a valid directory.")
    
    # Define the regex pattern to match hh_mm_ss at the start of the folder name
    pattern = re.compile(r'^(\d{2})_(\d{2})_(\d{2})')
    
    latest_time = None
    latest_folder = None
    
    for subfolder in parent_path.iterdir():
        if subfolder.is_dir():
            match = pattern.match(subfolder.name)
            if match:
                # Extract hours, minutes, and seconds
                hh, mm, ss = map(int, match.groups())
                try:
                    # Create a datetime.time object
                    current_time = datetime.strptime(subfolder.name[:8], "%H_%M_%S").time()
                except ValueError:
                    # Skip folders with invalid time formats
                    continue
                
                # Update latest_time and latest_folder if current_time is newer
                if (latest_time is None) or (current_time > latest_time):
                    latest_time = current_time
                    latest_folder = subfolder.name
    
    path = parent_dir + "/" + latest_folder

    return path
