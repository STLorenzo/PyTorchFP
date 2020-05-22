import os  # OS library
import json
import sys


# Creates a directory if it doesn't already exist
def create_dir(dir_name, debug=False):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        if debug:
            print("Directory ", dir_name, " Created ")
    else:
        if debug:
            print("Directory ", dir_name, " already exists")


def read_conf(relative_path="config/Project_conf.json"):
    abs_path = os.path.abspath(str(__file__) + "../../.." + relative_path)
    try:
        with open(abs_path, 'r') as json_file:
            return json.load(json_file)
    except Exception:
        print(f"Failed to open the path: {abs_path}")
        sys.exit(-1)
