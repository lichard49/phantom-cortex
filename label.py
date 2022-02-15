# General Imports
import json

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


"""
Takes a dictionary of experiment labels and appends it to the project json file

params: a dictionary of experiment labels
output: none
"""
def write_json(experiment):
    with open("test.json", "r+") as file:
        data = json.load(file)
        data["files"].append(experiment)
        file.seek(0)
        json.dump(data, file, indent = 4)
