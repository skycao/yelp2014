# run with ipython -pylab -i user_reviews.py

FILE_DIR = "../utilities/"

import sys
sys.path.insert(0, FILE_DIR)

import parse
import numpy
import matplotlib.pyplot as plt

user_file = "yelp_academic_dataset_user.json"

def get_user_review_counts():
    user_data = parse.parse_into_dicts(user_file);
    user_reviews = [user["review_count"] for user in user_data]
    return user_reviews

def user_reviews_hist():
    plt.hist(get_user_review_counts(), log=True)


