def make_shingle(document, shingle_length):
    assert type(document) is str, "must provide a string"
    assert type(shingle_length) is int, "must provide an int"

    words = document.split()
    return [words[i:i + shingle_length] for i in range(len(words) - shingle_length + 1)]
