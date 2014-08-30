# execute with python3 -i reviews.py run
# add a knn flag to do k-nearest-neighbors classification
# add a tfidf flag to compute the TF.IDF scores
# add a TIMED flag to have certain functions time themselves

from config import *
from yelp2014.utilities import parse
from yelp2014.utilities import document
from yelp2014.utilities import kNN
from yelp2014.utilities import output

TIMED = False

user_file = "yelp_academic_dataset_review.json"
SHINGLE_LENGTH = 4
NEAREST_NEIGHBOR_THRESHOLD = 5
partition = [[],[],[],[],[]]

def make_timed_utilities():
    """ Adds the timed decorator imported from decorators (from config) 
    to certain utility functions.

    """
    document.make_shingle = timed(document.make_shingle)
    document.make_set = timed(document.make_set)
    document.strip_punctuation = timed(document.strip_punctuation)
    document.term_freq = timed(document.term_freq)
    document.inverse_doc_freq = timed(document.inverse_doc_freq)
    document.tf_idf = timed(document.tf_idf)

def get_review_text(review_data):
    """ Constructs the list of review text corresponding to each review
    from the list review_data.

    review_data -- a list of dictionaries, each containing a "text" field
    """
    assert type(review_data) is list, "must provide a list (of dictionaries)"

    return [review["text"] for review in review_data]

def get_review_data():
    """ Constructs the list of objects that are obtained from parsing
    "yelp_academic_dataset_review.json".

    """
    return parse.parse_into_dicts(user_file)

def make_set_of_shingles(review_data):
    """ Constructs the set of shingles for each review in review_data. The shingle
    length is set by the global variable SHINGLE_LENGTH.

    review_data -- a list of dictionaries, each containing a "text" field
    """
    assert type(review_data) is list, "must provide a list (of dictionaries)"

    review_text = get_review_text(review_data)
    return [document.make_shingle(review, SHINGLE_LENGTH) for review in review_text]

def partition_by_rating():
    """ Constructs a list of 5 lists. The nested list at index i (0 <= i <= 4)
    contains all the reviews of star rating i + 1. 

    """
    review_data = get_review_data()
    partition = [[],[],[],[],[]]
    for data in review_data:
        partition[data["stars"] - 1].append(data)
    return partition

# too slow
def make_shingles_for_each_star():
    """ Constructs the set of shingles for all the reviews of each star rating. 

    """
    partition = partition_by_rating()
    return [make_set_of_shingles(category) for category in partition]

def make_set_of_words(review_data):
    """ Constructs the set of words for all the reviews in review_data.

    review_data -- a list of dictionaries, each containing a "text" field
    """
    assert type(review_data) is list, "must provide a list (of dictionaries)"
    
    review_text = get_review_text(review_data)
    return [document.make_set(review_text) for review in review_text]

def make_set_of_words_for_each_star():
    """ Constructs the set of words for all the reviews of each star rating.

    """
    partition = partition_by_rating()
    return [make_set_of_words(category) for category in partition]

def review_distance(review1, review2):
    """ Computes the jaccard distance between the sets of words found in
    review1 and review2. 

    Jacccard Distance of sets A and B is defined as
    1 - (|A \cap B| / |A \cup B|)

    review1 -- a dictionary containing a "text" field
    review2 -- a dictionary containing a "text" field
    """
    
    return document.jaccard_distance(document.make_set(review1["text"]), 
                                     document.make_set(review2["text"]))

def train_review_data():
    """ Trains a k-NN classifier on the first half of reviews found from each star
    rating category. 

    """
    data, labels = [], []
    for rating, category in enumerate(partition):
        length = len(category) // 2
        data += category[:length]
        # reviews in each category have their star rating as the label
        labels += [rating + 1] * length
    classifer = kNN.makeKNNClassifier(data, labels, NEAREST_NEIGHBOR_THRESHOLD, review_distance)
    if TIMED:
        return timed(classifier)
    else:
        return classifier
    
# Testing the data is too slow. The reason why is that the k-NN classifier is learned
# on 500,000 points. When a new data point needs to be classified, a priority heap of
# those 500,000 points is constructed. This construction takes too long. Other methods,
# such as locality-sensitive hashing, must be implemented to make k-NN feasible. 
def test_review_data():
    """ Tests the k-NN classifier trained by train_review_data() on the second half of 
    reviews found from each star rating category. Returns the list of absolute value differences
    observed at each test point. 

    """
    print("learning done")
    data, labels = [], []
    for rating, category in enumerate(partition):
        length = len(category) // 2
        data += category[length:]
        # reviews in each category have their star rating as the label
        labels += [rating + 1] * length
    print("beginning to test model")
    abs_dist = []
    for datum, label in zip(data, labels):
        abs_dist.append(abs(label - classifier(datum)))
    return abs_dist

def compute_tf_idf(review_data):
    """ Computes the tf_idf scores of the words found in the reviews
    of review_data. Returns a list of dictionaries. Each dictionary corresponds
    to a review, and maps words found in the review to the words' tf_idf scores.
    
    review_data -- a list of dictionaries, each containing a "text" field
    """
    reviews = [review["text"] for review in review_data]
    return document.tf_idf(reviews)
    
    
def construct_tf_idf_tables():
    """ Computes the tf_idf scores for each of the star rating categories.
    The individual categories are treated as different sets of documents, so
    5 sets of tf_idf scores are computed. A list of five lists is returned, where
    each nested list is a list of dictionaries. Each nested list corresponds to
    a star rating category, and each dictionary in the nested list corresponds to
    a review in that star rating category. The dictionaries maps words found in the
    corresponding reviews to the words' tf_idf scores.

    """
    return [compute_tf_idf(category) for category in partition]

if __name__ == "__main__":
    if "TIMED" in sys.argv:
        TIMED = True
        make_timed_utilities()
    if "run" in sys.argv:
        partition = partition_by_rating()
        print("partition done")
    if "knn" in sys.argv:
        result = test_review_data()
    if "tfidf" in sys.argv:
        tf_idf_scores = construct_tf_idf_tables()
