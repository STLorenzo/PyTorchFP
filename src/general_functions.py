import os  # OS library


# Creates a directory if it doesn't already exist
def create_dir(dir_name, debug=False):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        if debug:
            print("Directory ", dir_name, " Created ")
    else:
        if debug:
            print("Directory ", dir_name, " already exists")
