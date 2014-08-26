# code for kNN classifier taken from Jeremy Kun's Math \cap Programming blog
# visit http://jeremykun.com/2012/08/26/k-nearest-neighbors-and-handwritten-digit-classification/

import heapq

def makeKNNClassifier(data, labels, k, distance):
    """ Constructs a function that classifies data points by using
    k-nearest neighbors. 

    data -- the data points, should be a list
    labels -- the labels of the corresponding data points, should be a list
    k -- the number of neighbors to consider when classifying points
    distance -- the distance metric, should be a function that takes 2 parameters
    """
    def classify(x):
        closestPoints = heapq.nsmallest(k, enumerate(data),
                                        key=lambda y: distance(x, y[1]))
        closestLabels = [labels[i] for (i, pt) in closestPoints]
        return max(set(closestLabels), key=closestLabels.count)
 
    return classify
