import time

def timed(f):
    start = time.time()
    def wrapped(*args):
        result = f(*args)
        print("time taken by", f.__name__, ": ", time.time() - start)
        return result
    return wrapped
