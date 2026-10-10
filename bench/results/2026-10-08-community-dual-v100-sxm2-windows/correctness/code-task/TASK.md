Repair the three Python modules in this directory. Preserve their public signatures. Use only the Python standard library. Do not create other files or execute commands.

settings.py: parse_timeout(value, default=30) returns default unchanged only for None or a whitespace-only string. Accept nonnegative ints (including 0) and stripped ASCII decimal digit strings. Reject booleans, floats, negative ints, signed strings, non-ASCII digits, and all other types/strings with ValueError. The caller supplies a valid default.

cache.py: LRUCache(capacity) requires a positive integer, rejecting bool and other types with ValueError. get(key, default=None) returns the supplied default on a miss; a hit promotes the entry to most recently used, even when its stored value is None or 0. put updates or inserts and promotes the key. Evict the least recently used key only when inserting a new key would exceed capacity. Updating an existing key must not evict another key. All get/put operations should be O(1).

records.py: stable_unique(records) accepts any iterable of dictionaries. Keep the FIRST record for each id, preserve input order, and do not mutate input dictionaries. Every record missing the id key is retained independently. An id of None is a real id and deduplicates normally. ids are hashable. Returned records retain their original object identity.

Read the files, make the repairs, then provide a brief summary. Do not modify TASK.md.
