def write_to_file(data, file_name, dir_name='.'):
    """ Writes each element of data to a file.

    data -- a list
    file_name -- name of the file to be written to
    dir_name -- name of the directory that should hold the file
    """
    assert type(data) is list, "data must be a list"
    
    new_file = open(dir_name + "/" + file_name, 'w')
    for item in data:
        new_file.write(str(item) + "\n")
    new_file.close()
