import time

def timed(f):
    def wrapped(*args):
        start = time.time()
        result = f(*args)
        print("time taken by {0}: {1}".format(f.__name__, time.time() - start))
        return result
    return wrapped
