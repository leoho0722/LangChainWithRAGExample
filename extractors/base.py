class BaseExtractor:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def extract(self, *args, **kwargs):
        raise NotImplementedError("You must implement the `extract` method")

    def __call__(self, *args, **kwargs):
        return self.extract(*args, **kwargs)
