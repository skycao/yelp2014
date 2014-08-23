from config import *
from yelp2014.utilities import parse
from yelp2014.utilities import document
from yelp2014.utilities import kNN

user_file = "yelp_academic_dataset_review.json"
SHINGLE_LENGTH = 4
NEAREST_NEIGHBOR_THRESHOLD = 5
partition = [[],[],[],[],[]]

def get_review_text(review_data):
    return [review["text"] for review in review_data]

def get_review_data():
    return parse.parse_into_dicts(user_file)

def get_sorted_review_data():
    return sorted(get_review_data(), key = lambda review: review["stars"])

def make_set_of_shingles(review_data):
    review_text = get_review_text(review_data)
    return [document.make_shingle(review, SHINGLE_LENGTH) for review in review_text]

# returns a list of lists
# each nested list contains all the reviews for a certain rating
def partition_by_rating():
    review_data = get_review_data()
    partition = [[],[],[],[],[]]
    for data in review_data:
        partition[data["stars"] - 1].append(data)
    return partition

# too slow
def make_shingles_for_each_star():
    partition = partition_by_rating()
    return [make_set_of_shingles(category) for category in partition]

def make_set_of_words(review_data):
    review_text = get_review_text(review_data)
    return [document.make_set(review_text) for review in review_text]

def make_set_of_words_for_each_star():
    partition = partition_by_rating()
    return [make_set_of_words(category) for category in partition]

def review_distance(review1, review2):
    return document.jaccard_distance(document.make_set(review1["text"]), 
                                     document.make_set(review2["text"]))

# trains on first half of reviews from each star rating category
def train_review_data():
    data, labels = [], []
    for rating, category in enumerate(partition):
        length = len(category) / 2
        data += category[:length]
        # reviews in each category have their star rating as the label
        labels.append([rating + 1] * length)
    return kNN.makeKNNClassifier(data, labels, NEAREST_NEIGHBOR_THRESHOLD, review_distance)
    
# tests on second half of reviews from each star rating category
def test_review_data():
    classifier = train_review_data()
    data, labels = [], []
    for rating, category in enumerate(partition):
        length = len(category) / 2
        data += category[length:]
        # reviews in each category have their star rating as the label
        labels.append([rating + 1] * length)
    abs_dist = 0
    for datum, label in zip(data, labels):
        abs_dist += abs(label - classifier(data))

