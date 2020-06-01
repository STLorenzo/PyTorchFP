import os  # OS library
import json
import sys


# Creates a directory if it doesn't already exist
def create_dir(dir_path, debug=False):
    """
    Creates a directory if doesn't already exist in specified path

    Parameters
    ----------
    dir_path : Path
        path to the directory
    debug : bool
        flag that controls if the result is printed
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        if debug:
            print("Directory ", dir_path, " Created ")
    else:
        if debug:
            print("Directory ", dir_path, " already exists")


def read_conf(relative_path="config/Project_conf.json"):
    """
    Functions that reads a json configuration file

    Parameters
    ----------
    relative_path : Path
        path to the file

    Returns
    -------
    the data loaded

    Raises
    ------
    Exception if there is an error and exits the program
    """
    abs_path = os.path.abspath(str(__file__) + "../../.." + relative_path)
    try:
        with open(abs_path, 'r') as json_file:
            return json.load(json_file)
    except Exception:
        print(f"Failed to open the path: {abs_path}")
        sys.exit(-1)
