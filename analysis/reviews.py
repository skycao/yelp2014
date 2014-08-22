from config import *

import parse
import shingle

user_file = "yelp_academic_dataset_review.json"
SHINGLE_LENGTH = 4

def make_set_of_shingles(review_data):
    review_text = [review["text"] for review in review_data]
    return [shingle.make_shingle(review, SHINGLE_LENGTH) for review in review_text]

def get_review_data():
    return parse.parse_into_dicts(user_file)

def get_sorted_review_data():
    return sorted(get_review_data(), key = lambda review: review["stars"])

# returns a list of lists
# each nested list contains all the reviews for a certain rating
def partition_by_rating():
    review_data = get_review_data()
    partition = [[],[],[],[],[]]
    for data in review_data:
        partition[data["stars"] - 1].append(data)
    return partition

def make_shingles_for_each_star():
    partition = partition_by_rating()
    return [make_set_of_shingles(category) for category in partition]


    
