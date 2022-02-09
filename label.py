# General Imports
import json
from brainflow import DataFilter

"""
Parses json file into a dictionary

params: a valid json filename
output: returns a dictionary where the keys are the filenames
        and the values are the associated labels
"""
def import_json(json_filename):
    json_data = open(json_filename, "r")
    data = json.loads(json_data.read())

    files = data["files"]
    file_dict = {}
    for f in files:
        file_dict[f["filename"]] = f["labels"]
    json_data.close()
    return file_dict

def load_data(filename):
    readdata = DataFilter.read_file(filename)
    return readdata