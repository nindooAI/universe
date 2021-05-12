import os

def make_dirs(directories_list:list):
    for directory in directories_list:
        if not os.path.exists(directory):
            os.makedirs(directory)