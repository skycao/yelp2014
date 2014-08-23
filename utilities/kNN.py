# code for kNN classifier taken from Jeremy Kun's Math \cap Programming blog
# visit http://jeremykun.com/2012/08/26/k-nearest-neighbors-and-handwritten-digit-classification/

import heapq

def makeKNNClassifier(data, labels, k, distance):
    def classify(x):
        closestPoints = heapq.nsmallest(k, enumerate(data),
                                        key=lambda y: distance(x, y[1]))
        closestLabels = [labels[i] for (i, pt) in closestPoints]
        return max(set(closestLabels), key=closestLabels.count)
 
    return classify
