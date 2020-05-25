import json

def dict_raise_on_duplicates(ordered_pairs):
    """Reject duplicate keys."""
    d = {}
    for k, v in ordered_pairs:
        if k in d:
           raise ValueError("duplicate key: %r" % (k,))
        else:
           d[k] = v
    return d


class Params():
    """Class that loads hyperparameters from a json file or from a dict.
    Example:
    ```
    params = Params(json_path or dict)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, inp):
        if isinstance(inp, str):
            with open(inp) as f:
                params = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
                self.__dict__.update(params)
        elif isinstance(inp, dict):
            self.__dict__.update(inp)
        else:
            raise TypeError ("input not understood")

    def save(self, json_path):
        try:
            with open(json_path, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        except TypeError:
            for k, v in self.__dict__.items():
                d = dict()
                d[k] = v
                try:
                    with open(json_path, 'w') as f:
                        json.dump(d, f, indent=4)
                except TypeError:
                    types = set([type(v)])
                    if hasattr(v,'__iter__'):
                        for e in v:
                            types.add(type(e))
                    print('json type error for key, value {} pair: {}:{}'.format(type(v),k,v))
                    print('types found', types)
            raise

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
