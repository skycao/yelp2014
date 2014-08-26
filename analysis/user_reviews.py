# run with ipython --matplotlib auto -i user_reviews.py

from config import *

import parse
import numpy
import pylab
import matplotlib.pyplot as plt

user_file = "yelp_academic_dataset_user.json"

def get_user_data():
    """ Parses the yelp user data file and constructs
    the list of dictionary objects.

    """
    return parse.parse_into_dicts(user_file)

def get_user_review_counts():
    """ Constructs the list of review counts for each user.

    """
    user_data = get_user_data()
    user_reviews = [user["review_count"] for user in user_data]
    return user_reviews

def user_reviews_hist():
    """ Plots the reviews counts for each user as a histogram. 
    The y-axis is log-scaled.
    
    """
    data = plt.hist(get_user_review_counts(), log=True)
    #y, x = data[0].tolist(), data[1].tolist()
    #y.append(0)
    #plt.plot(x, y, 'y', scalex=False, scaley=False)
    return data

if __name__ == "__main__":
    data = user_reviews_hist()
