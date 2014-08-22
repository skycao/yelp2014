# run with ipython --matplotlib auto -i user_reviews.py

FILE_DIR = "../utilities/"

import sys
sys.path.insert(0, FILE_DIR)

import parse
import numpy
import pylab
import matplotlib.pyplot as plt

user_file = "yelp_academic_dataset_user.json"

def get_user_data():
    return parse.parse_into_dicts(user_file)

def get_user_review_counts():
    user_data = get_user_data()
    user_reviews = [user["review_count"] for user in user_data]
    return user_reviews

def user_reviews_hist():
    data = plt.hist(get_user_review_counts(), log=True)
    #y, x = data[0].tolist(), data[1].tolist()
    #y.append(0)
    #plt.plot(x, y, 'y', scalex=False, scaley=False)
    return data

if __name__ == "__main__":
    data = user_reviews_hist()
