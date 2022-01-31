import json

# to parse json file 

# open json file
def import_json(json_filename):
    json_data = open(json_filename, "r")
    data = json.loads(json_data.read())

    files = data["files"]
    file_dict = {}
    for f in files:
        file_dict[f["filename"]] = f["labels"]
    json_data.close()
    return file_dict
