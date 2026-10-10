from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key, default=None):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return default

    def put(self, key, value):
        if key in self.data:
            self.data[key] = value
            self.data.move_to_end(key)
        else:
            if len(self.data) >= self.capacity:
                self.data.popitem(last=False)
            self.data[key] = value
