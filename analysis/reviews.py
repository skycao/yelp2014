# execute with python3 -i reviews.py run

from config import *
from yelp2014.utilities import parse
from yelp2014.utilities import document
from yelp2014.utilities import kNN

user_file = "yelp_academic_dataset_review.json"
SHINGLE_LENGTH = 4
NEAREST_NEIGHBOR_THRESHOLD = 5
partition = [[],[],[],[],[]]

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
    return kNN.makeKNNClassifier(data, labels, NEAREST_NEIGHBOR_THRESHOLD, review_distance)
    
# Testing the data is too slow. The reason why is that the k-NN classifier is learned
# on 500,000 points. When a new data point needs to be classified, a priority heap of
# those 500,000 points is constructed. This construction takes too long. Other methods,
# such as locality-sensitive hashing, must be implemented to make k-NN feasible. 
def test_review_data():
    """ Tests the k-NN classifier trained by train_review_data() on the second half of 
    reviews found from each star rating category. Returns the list of absolute value differences
    observed at each test point. 

    """
    classifier = train_review_data()
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

if __name__ == "__main__":
    if len(sys.argv) > 1 and str(sys.argv[1]) == "run":
        partition = partition_by_rating()
        print("partition done")
        result = test_review_data()
