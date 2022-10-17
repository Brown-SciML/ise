import yaml
import os

file_dir = os.path.dirname(os.path.realpath(__file__))

def get_configs():
    # Loads configuration file in the repo and formats it as a dictionary.
    try:
        with open(f'{file_dir}/config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    except FileNotFoundError:  
        try:   # depends on where you're calling it from...
            with open('./config.yaml') as c:
                data = yaml.load(c, Loader=yaml.FullLoader)
        except FileNotFoundError:
            with open('config.yaml') as c:
                data = yaml.load(c, Loader=yaml.FullLoader)
    return data


def check_input(input, options):
    # simple assert that input is in the designated options (readability purposes only)
    input = input.lower()
    assert input in options, f"input must be in {options}, received {input}"

def get_all_filepaths(path, filetype):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    if filetype.lower() != 'all':
        listOfFiles = [file for file in listOfFiles if file.endswith(filetype)]
    return listOfFiles