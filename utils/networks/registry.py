class Registry(object):
    def __init__(self, name):
        # name of register
        self._name = name
        self._name_method_map = dict()

    def __setitem__(self, key, value):
        if not callable(value):
            raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
        if key is None:
            key = value.__name__
        if key in self._name_method_map:
            raise Exception(f"Key {key} already in registry.")
        self._name_method_map[key] = value

    def __getitem__(self, item):
        return self._name_method_map[item]

    def __contains__(self, item):
        return item in self._name_method_map

    def register(self, obj=None):
        def add(key, value):
            self[key] = value
            return value

        if callable(obj):
            return add(None, obj)
        else:
            return lambda x: add(obj, x)

    def get_all_keys(self):
        return self._name_method_map.keys()


ModelRegistry = Registry("register")
