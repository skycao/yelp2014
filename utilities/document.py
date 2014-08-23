import string

# constructs the set of all k-shingles for the given document
# where k is shingle_length
# the constructed shingles are based on words, not characters
def make_shingle(document, shingle_length):
    assert type(document) is str, "must provide a string"
    assert type(shingle_length) is int, "must provide an int"

    words = document.split()
    return [words[i:i + shingle_length] for i in range(len(words) - shingle_length + 1)]

# constructs the set of words seen in the document
def make_set(document):
    assert type(document) is str, "must provide a string"
    return set(strip_punctuation(document).split())

# maps all punctuation to empty string
# used by str.translate to map all punctuation to empty string
# thereby stripping str of all punctuation
punc_map = dict([(punc, "") for punc in string.punctuation])

def strip_punctuation(sentence):
    assert type(sentence) is str, "must provide a string"
    return sentence.translate(sentence.maketrans(punc_map))

# defined as 1 - ( |set1 \cap set2| / |set1 \cup \set2] )
def jaccard_distance(set1, set2):
    assert type(set1) is set, "first argument must be a set"
    assert type(set2) is set, "second argument must be a set"
    # could use the built in intersect and union methods
    # but this way is likely faster
    intersection, union = 0, len(set1)
    for elem in set2:
        if elem in set1:
            intersection += 1
        else:
            union += 1
    return 1 - float(intersection) / union
    
    
