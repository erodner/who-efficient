import time

class Timer:
    def elapsed(self):
        return (time.time() - self.tstart)

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %.5f' % self.elapsed()


