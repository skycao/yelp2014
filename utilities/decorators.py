import time

def timed(f):
    """ Returns a function that executes f and prints out how long 
    the execution took.

    f -- a function, can have any number of parameters
    """
    def wrapped(*args):
        start = time.time()
        result = f(*args)
        print("time taken by {0}: {1}".format(f.__name__, time.time() - start))
        return result
    return wrapped
