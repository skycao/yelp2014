import string
import math

def make_shingle(document, shingle_length):
    """ Constructs the set of all k-shingles for the given document,
    where k is shingle_length. The constructed shingles are based on words,
    not characters.

    document -- a string, presumably one or more sentences
    shingle_length -- length of the constructed shingles
    """
    assert type(document) is str, "must provide a string"
    assert type(shingle_length) is int, "must provide an int"

    words = document.split()
    return [words[i:i + shingle_length] for i in range(len(words) - shingle_length + 1)]

def make_set(document):
    """ Constructs the set of words seen in the document.

    document -- a string, presumably one or more sentences
    """
    assert type(document) is str, "must provide a string"
    return set(strip_punctuation(document).split())

# maps all punctuation to empty string
# can be used by str.translate to strip a string of its punctuation
punc_map = dict([(punc, "") for punc in string.punctuation])

def strip_punctuation(sentence):
    """ Strips sentence of its punctuation.

    sentence -- a string
    """
    assert type(sentence) is str, "must provide a string"
    return sentence.translate(sentence.maketrans(punc_map))

def jaccard_distance(set1, set2):
    """ Computes the jaccard distance between set1 and set2.

    Jaccard distance is defined as
    1 - ( |set1 \cap set2| / |set1 \cup \set2] )

    set1 -- a set
    set2 -- a set
    """
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

def term_freq(document):
    """ Constructs a dictionary where the keys are the terms appearing
    in document, and the values are the corresponding term frequencies.

    The TF (term frequency) of a term i in a document j is defined as:
    n_i / max_j
    where n_i is the number of times i appears in j, and max_j is
    the maximum number of times any term appears in j.

    document -- a str, presumably one or more sentences
    """
    assert type(document) is str, "must provide a str"

    document = document.lower()
    term_freq_dict = {}
    terms = strip_punctuation(document).split()

    max_freq = 0
    for term in terms:
        if term in term_freq_dict:
            term_freq_dict[term] += 1
        else:
            term_freq_dict[term] = 1
        if term_freq_dict[term] > max_freq:
            max_freq = term_freq_dict[term]

    if max_freq > 0:
        for term in term_freq_dict:
            term_freq_dict[term] /= max_freq
            
    return term_freq_dict

def inverse_doc_freq(term_freq_dicts):
    """ Calculates the inverse document frequency of each of the terms 
    found in a given set of documents.

    The definition of IDF_i for a given term i is
    log_2 (N / n_i) where N is the total number of documents
    and n_i is the number of documents where the i occurs

    term_freq_dicts -- a list of dictionaries, of the same format as those
    returned by term_freq()
    """
    assert type(term_freq_dicts) is list, "must provide a list (of dictionaries)"
    start = time.time()

    inv_doc_freq_dict = {}
    for freq_dict in term_freq_dicts:
        for term in freq_dict:
            if term in inv_doc_freq_dict:
                inv_doc_freq_dict[term] += 1
            else:
                inv_doc_freq_dict[term] = 1

    num_docs = len(term_freq_dicts)
    for term in inv_doc_freq_dict:
        inv_doc_freq_dict[term] = math.log(num_docs / float(inv_doc_freq_dict[term]), 2)
    
    print("inverse_doc_freq took %d", time.time() - start)
    return inv_doc_freq_dict

def tf_idf(documents):
    """ Constructs a list of dictionaries. Each dictionary corresponds
    to a given document. The keys of the dictionaries are the terms
    that appear in the corresponding document, and the values are the
    TF.IDF scores of the terms. 

    Given the TF and IDF of a term i, the TF.IDF score is defined as
    TF * IDF

    documents -- a list of documents, i.e. strings
    """
    assert type(documents) is list, "must provide a list (of strings)"

    term_freq_dicts = [term_freq(doc) for doc in documents]
    inv_doc_freq_dict = inverse_doc_freq(term_freq_dicts)
    
    # for each table (corresponding to a document)
    # create a dictionary with keys as the terms in the table
    # and values as the TF.IDF values of the terms in the table
    tf_idf_dict = [dict([(term, term_freq[term] * inv_doc_freq_dict[term]) for term in term_freq]) 
                   for term_freq in term_freq_dicts]

    return tf_idf_dicts
    
