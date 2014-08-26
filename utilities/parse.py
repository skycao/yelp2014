import json

FILE_DIR = "../data/"

def parse_into_dicts(json_file, file_dir=FILE_DIR):
    """ Parses a JSON-formatted file into a list of dictionary objects
    
    json_file -- the name of the file
    file_dir -- the directory where the file resides
    """
    json_data = open(file_dir + json_file)
    decoder = json.JSONDecoder()
    data = []
    line = json_data.readline()
    while line:
        data.append(decoder.decode(line))
        line = json_data.readline()
    json_data.close()
    return data
