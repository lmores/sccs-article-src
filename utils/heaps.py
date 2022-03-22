from .misc import SRG


class MinHeap:

    def __init__(self, kv_pairs=None):
        _keys = self._keys = []
        _values = self._values = []
        _indexes = self._indexes = {}

        if kv_pairs:
            for k,v in kv_pairs:
                _indexes[k] = len(_keys)
                _keys.append(k)
                _values.append(v)

            self._heapify()

    ##########################################################################
    ### Magic methods
    ##########################################################################

    def __bool__(self):
        return bool(self._keys)

    def __contains__(self, key):
        return key in self._indexes

    def __eq__(self, other):
        if not isinstance(other, MinHeap):
            return False

        _keys, _okeys = self._keys, other._keys
        _values, _ovalues = self._values, other._values
        _indexes, _oindexes = self._indexes, other._indexes

        if len(_keys) != len(_okeys):
            return False

        for key in _keys:
            pos, opos = _indexes[key], _oindexes.get(key, None)
            if opos is None or _values[pos] != _ovalues[opos]:
                return False

        return True

    def __len__(self):
        return len(self._keys)

    def __str__(self):
        cls = type(self)
        return (f"<{cls.__module__}.{cls.__qualname__} object "
                f"at 0x{id(self):x}, size: {len(self._keys)}>")

    def __repr__(self):
        cls = type(self)
        return (f"<{cls.__module__}.{cls.__qualname__} object "
                f"at 0x{id(self):x}, size: {len(self._keys)}>")

    def __deepcopy__(self, memo=None):
        cp = MinHeap()
        cp._keys = self._keys.copy()
        cp._values = self._values.copy()
        cp._indexes =  self._indexes.copy()
        return cp

    ##########################################################################
    ### Public methods
    ##########################################################################

    def push(self, key, value):
        """Push key and value onto heap, maintaining the heap invariant."""
        _indexes = self._indexes
        if key in _indexes:
            raise KeyError(f"Key {key} is already present inside the heap")

        pos = _indexes[key] = len(self._keys)
        self._keys.append(key)
        self._values.append(value)

        self._siftdown(0, pos)

    def peek(self):
        return self._keys[0], self._values[0]

    def peek_key(self):
        return self._keys[0]

    def peek_value(self):
        return self._values[0]

    def pop(self):
        """Pop the smallest item off the heap and restore the heap invariant."""
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        # Raise appropriate IndexError if heap is empty
        last_key, last_value = _keys.pop(), _values.pop()

        if _keys:
            smallest_key, smallest_value = _keys[0], _values[0]
            del _indexes[smallest_key]

            _keys[0], _values[0] = last_key, last_value
            _indexes[last_key] = 0
            self._siftup(0)

            return smallest_key, smallest_value
        else:
            del _indexes[last_key]

        return last_key, last_value

    def update(self, key, new_value):
        # Raise appropriate KeyError if 'key' is missing
        pos = self._indexes[key]
        old_value = self._values[pos]
        self._values[pos] = new_value

        if new_value > old_value:
            self._siftup(pos)
        elif new_value < old_value:
            self._siftdown(0, pos)

    def update_delta(self, key, delta):
        # Raise appropriate KeyError if key is missing
        pos = self._indexes[key]
        new_value = self._values[pos] = self._values[pos] + delta

        if delta > 0:
            self._siftup(pos)
        elif delta < 0:
            self._siftdown(0, pos)

        return new_value

    def delete(self, key):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        # Raise appropriate KeyError if 'key' is missing
        pos = _indexes[key]

        if pos < len(_keys) - 1:
            value = _values[pos]
            _keys[pos], _values[pos] = _keys.pop(), _values.pop()
            del _indexes[key]

            if pos > 0 and _values[pos] < _values[(pos - 1) >> 1]:  # parent_pos
                self._siftdown(0, pos)
            else:
                self._siftup(pos)

            return value
        else:
            _keys.pop()
            del _indexes[key]
            return _values.pop()

    def discard(self, key):
        try:
            return self.delete(key)
        except KeyError:
            return None

    def replace(self, key, value):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        if key in _indexes:
            raise KeyError(f"Key {key} is already present inside the heap")

        smallest_key, smallest_value = _keys[0], _values[0]
        del _indexes[smallest_key]

        _keys[0], _values[0] = key, value
        _indexes[key] = 0
        self._siftup(0)

        return smallest_key, smallest_value

    def pushpop(self, key, value):
        _keys = self._keys
        _values = self._values

        if _values and _values[0] < value:
            key, value, _keys[0], _values[0] = _keys[0], _values[0], key, value
            del self._indexes[key]
            self._indexes[key] = 0
            self._siftup(0)

        return key, value

    def _heapify(self):
        n = len(self._keys)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        _siftup = self._siftup
        for i in reversed(range(n//2)):
            _siftup(i)

    # 'heap' is a heap at all indices >= 'startpos', except possibly for 'pos'.
    # 'pos' is the index of a leaf with a possibly out-of-order value.
    # Restore the heap invariant.
    def _siftdown(self, start_pos, pos):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        key, value = _keys[pos], _values[pos]
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent_key, parent_value = _keys[parent_pos], _values[parent_pos]

            if value >= parent_value:
                break

            _keys[pos], _values[pos] = parent_key, parent_value
            _indexes[parent_key] = pos
            pos = parent_pos

        _keys[pos], _values[pos] = key, value
        _indexes[key] = pos

    # The child indices of heap index pos are already heaps, and we want to make
    # a heap at index pos too.  We do this by bubbling the smaller child of
    # pos up (and so on with that child's children, etc) until hitting a leaf,
    # then using _siftdown to move the oddball originally at index pos into place.
    #
    # We *could* break out of the loop as soon as we find a pos where newitem <=
    # both its children, but turns out that's not a good idea, and despite that
    # many books write the algorithm that way.  During a heap pop, the last array
    # element is sifted in, and that tends to be large, so that comparing it
    # against values starting from the root usually doesn't pay (= usually doesn't
    # get us out of the loop early).  See Knuth, Volume 3, where this is
    # explained and quantified in an exercise.
    #
    # Cutting the # of comparisons is important, since these routines have no
    # way to extract "the priority" from an array element, so that intelligence
    # is likely to be hiding in custom comparison methods, or in array elements
    # storing (priority, record) tuples.  Comparisons are thus potentially
    # expensive.
    #
    # On random arrays of length 1000, making this change cut the number of
    # comparisons made by heapify() a little, and those made by exhaustive
    # heappop() a lot, in accord with theory.  Here are typical results from 3
    # runs (3 just to demonstrate how small the variance is):
    #
    # Compares needed by heapify     Compares needed by 1000 heappops
    # --------------------------     --------------------------------
    # 1837 cut to 1663               14996 cut to 8680
    # 1855 cut to 1659               14966 cut to 8678
    # 1847 cut to 1660               15024 cut to 8703
    #
    # Building the heap by using heappush() 1000 times instead required
    # 2198, 2148, and 2219 compares:  heapify() is more efficient, when
    # you can use it.
    #
    # The total compares needed by list.sort() on the same lists were 8627,
    # 8627, and 8632 (this should be compared to the sum of heapify() and
    # heappop() compares):  list.sort() is (unsurprisingly!) more efficient
    # for sorting.
    def _siftup(self, pos):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        start_pos, end_pos = pos, len(_keys)
        key, value = _keys[pos], _values[pos]

        child_pos = 2 * pos + 1    # leftmost child position
        while child_pos < end_pos:
            # Set child_pos to the index of the child with the smaller value.
            right_pos = child_pos + 1
            if right_pos < end_pos and _values[child_pos] > _values[right_pos]:
                child_pos = right_pos

            # Move the smaller child up.
            _keys[pos], _values[pos] = _keys[child_pos], _values[child_pos]
            _indexes[_keys[pos]] = pos
            pos = child_pos
            child_pos = 2 * pos + 1

        # The leaf at pos is empty now.  Put value there, and bubble it up
        # to its final resting place (by sifting its parents down).
        _keys[pos], _values[pos] = key, value
        _indexes[_keys[pos]] = pos
        self._siftdown(start_pos, pos)

    def pop_random(self):
        pos = SRG.randint(0, len(self._keys) - 1)
        key, value = self._keys[pos], self._values[pos]
        self.delete(key)

        return key, value

    def set_state(self, keys, values, indexes):
        self._keys = keys
        self._values = values
        self._indexes = indexes


class MaxHeap:

    def __init__(self, kv_pairs=None):
        _keys = self._keys = []
        _values = self._values = []
        _indexes = self._indexes = {}

        if kv_pairs:
            for k,v in kv_pairs:
                _indexes[k] = len(_keys)
                _keys.append(k)
                _values.append(v)

            self._heapify()

    ##########################################################################
    ### Magic methods
    ##########################################################################

    def __bool__(self):
        return bool(self._keys)

    def __contains__(self, key):
        return key in self._indexes

    def __eq__(self, other):
        if not isinstance(other, MaxHeap):
            return False

        _keys, _okeys = self._keys, other._keys
        _values, _ovalues = self._values, other._values
        _indexes, _oindexes = self._indexes, other._indexes

        if len(_keys) != len(_okeys):
            return False

        for key in _keys:
            pos, opos = _indexes[key], _oindexes.get(key, None)
            if opos is None or _values[pos] != _ovalues[opos]:
                return False

        return True

    def __len__(self):
        return len(self._keys)

    def __str__(self):
        cls = type(self)
        return (f"<{cls.__module__}.{cls.__qualname__} object "
                f"at 0x{id(self):x}, size: {len(self._keys)}>")

    def __repr__(self):
        cls = type(self)
        return (f"<{cls.__module__}.{cls.__qualname__} object "
                f"at 0x{id(self):x}, size: {len(self._keys)}>")

    def __deepcopy__(self, memo=None):
        cp = MaxHeap()
        cp._keys = self._keys.copy()
        cp._values = self._values.copy()
        cp._indexes =  self._indexes.copy()
        return cp

    ##########################################################################
    ### Public methods
    ##########################################################################

    def push(self, key, value):
        _indexes = self._indexes
        if key in _indexes:
            raise KeyError(f"Key {key} is already present inside the heap")

        pos = _indexes[key] = len(self._keys)
        self._keys.append(key)
        self._values.append(value)

        self._siftdown(0, pos)

    def peek(self):
        return self._keys[0], self._values[0]

    def peek_key(self):
        return self._keys[0]

    def peek_value(self):
        return self._values[0]

    def pop(self):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        # Raise appropriate IndexError if heap is empty
        last_key, last_value = _keys.pop(), _values.pop()

        if _keys:
            largest_key, largest_value = _keys[0], _values[0]
            del _indexes[largest_key]

            _keys[0], _values[0] = last_key, last_value
            _indexes[last_key] = 0
            self._siftup(0)

            return largest_key, largest_value
        else:
            del _indexes[last_key]

        return last_key, last_value

    def update(self, key, new_value):
        # Raise appropriate KeyError if 'key' is missing
        pos = self._indexes[key]
        old_value = self._values[pos]
        self._values[pos] = new_value

        if new_value < old_value:
            self._siftup(pos)
        elif new_value > old_value:
            self._siftdown(0, pos)

    def update_delta(self, key, delta):
        # Raise appropriate KeyError if key is missing
        pos = self._indexes[key]
        new_value = self._values[pos] = self._values[pos] + delta

        if delta < 0:
            self._siftup(pos)
        elif delta > 0:
            self._siftdown(0, pos)

        return new_value

    def delete(self, key):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        # Raise appropriate KeyError if 'key' is missing
        pos = _indexes[key]

        if pos < len(_keys) - 1:
            value = _values[pos]
            _keys[pos], _values[pos] = _keys.pop(), _values.pop()
            del _indexes[key]

            if pos > 0 and _values[pos] > _values[(pos - 1) >> 1]:  # parent_pos
                self._siftdown(0, pos)
            else:
                self._siftup(pos)

            return value
        else:
            _keys.pop()
            del _indexes[key]
            return _values.pop()

    def discard(self, key):
        try:
            return self.delete(key)
        except KeyError:
            return None

    def replace(self, key, value):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        if key in _indexes:
            raise KeyError(f"Key {key} is already present inside the heap")

        largest_key, largest_value = _keys[0], _values[0]
        del _indexes[largest_key]

        _keys[0], _values[0] = key, value
        _indexes[key] = 0
        self._siftup(0)

        return largest_key, largest_value

    def pushpop(self, key, value):
        _keys = self._keys
        _values = self._values

        if _values and _values[0] > value:
            key, value, _keys[0], _values[0] = _keys[0], _values[0], key, value
            del self._indexes[key]
            self._indexes[key] = 0
            self._siftup(0)

        return key, value

    def _heapify(self):
        n = len(self._keys)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        _siftup = self._siftup
        for i in reversed(range(n//2)):
            _siftup(i)

    # 'heap' is a heap at all indices >= 'startpos', except possibly for 'pos'.
    # 'pos' is the index of a leaf with a possibly out-of-order value.
    # Restore the heap invariant.
    def _siftdown(self, start_pos, pos):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        key, value = _keys[pos], _values[pos]
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent_key, parent_value = _keys[parent_pos], _values[parent_pos]

            if value <= parent_value:
                break

            _keys[pos], _values[pos] = parent_key, parent_value
            _indexes[parent_key] = pos
            pos = parent_pos

        _keys[pos], _values[pos] = key, value
        _indexes[key] = pos

    # The child indices of heap index pos are already heaps, and we want to make
    # a heap at index pos too.  We do this by bubbling the smaller child of
    # pos up (and so on with that child's children, etc) until hitting a leaf,
    # then using _siftdown to move the oddball originally at index pos into place.
    #
    # We *could* break out of the loop as soon as we find a pos where newitem <=
    # both its children, but turns out that's not a good idea, and despite that
    # many books write the algorithm that way.  During a heap pop, the last array
    # element is sifted in, and that tends to be large, so that comparing it
    # against values starting from the root usually doesn't pay (= usually doesn't
    # get us out of the loop early).  See Knuth, Volume 3, where this is
    # explained and quantified in an exercise.
    #
    # Cutting the # of comparisons is important, since these routines have no
    # way to extract "the priority" from an array element, so that intelligence
    # is likely to be hiding in custom comparison methods, or in array elements
    # storing (priority, record) tuples.  Comparisons are thus potentially
    # expensive.
    #
    # On random arrays of length 1000, making this change cut the number of
    # comparisons made by heapify() a little, and those made by exhaustive
    # heappop() a lot, in accord with theory.  Here are typical results from 3
    # runs (3 just to demonstrate how small the variance is):
    #
    # Compares needed by heapify     Compares needed by 1000 heappops
    # --------------------------     --------------------------------
    # 1837 cut to 1663               14996 cut to 8680
    # 1855 cut to 1659               14966 cut to 8678
    # 1847 cut to 1660               15024 cut to 8703
    #
    # Building the heap by using heappush() 1000 times instead required
    # 2198, 2148, and 2219 compares:  heapify() is more efficient, when
    # you can use it.
    #
    # The total compares needed by list.sort() on the same lists were 8627,
    # 8627, and 8632 (this should be compared to the sum of heapify() and
    # heappop() compares):  list.sort() is (unsurprisingly!) more efficient
    # for sorting.
    def _siftup(self, pos):
        _keys = self._keys
        _values = self._values
        _indexes = self._indexes

        start_pos, end_pos = pos, len(_keys)
        key, value = _keys[pos], _values[pos]

        child_pos = 2 * pos + 1    # leftmost child position
        while child_pos < end_pos:
            # Set child_pos to the index of the child with the larger value.
            right_pos = child_pos + 1
            if right_pos < end_pos and _values[child_pos] < _values[right_pos]:
                child_pos = right_pos

            # Move the larger child up.
            _keys[pos], _values[pos] = _keys[child_pos], _values[child_pos]
            _indexes[_keys[pos]] = pos
            pos = child_pos
            child_pos = 2 * pos + 1

        # The leaf at pos is empty now.  Put value there, and bubble it up
        # to its final resting place (by sifting its parents down).
        _keys[pos], _values[pos] = key, value
        _indexes[_keys[pos]] = pos
        self._siftdown(start_pos, pos)

    def pop_random(self):
        pos = SRG.randint(0, len(self._keys) - 1)
        key, value = self._keys[pos], self._values[pos]
        self.delete(key)

        return key, value

    def set_state(self, keys, values, indexes):
        self._keys = keys
        self._values = values
        self._indexes = indexes


def _check_min_heap_consistency(h: MinHeap):
    keys = h._keys
    values = h._values
    indexes = h._indexes

    #print(keys, values, indexes)

    assert len(keys) == len(values) == len(indexes)

    for i,v in enumerate(values):
        if 2*i + 1 < len(values):
            assert v <= values[2*i + 1], \
                f"{v} = heap[{i}] > heap[{2*i + 1}] = {values[2*i + 1]}, {values}"
        if 2*i + 2 < len(values):
            assert v <= values[2*i + 2], \
                f"{v} = heap[{i}] > heap[{2*i + 2}] = {values[2*i + 2]}, {values}"

    indexes_keys = set(h._indexes.keys())
    expected_indexes_keys = set(keys)
    assert indexes_keys == expected_indexes_keys, \
        f"{indexes_keys} != {expected_indexes_keys}"

    indexes_positions = set(h._indexes.values())
    expected_indexes_positions = set(range(len(keys)))
    assert indexes_positions == expected_indexes_positions, \
        f"{indexes_positions} != {expected_indexes_positions}"

    print('Min heap: OK!')


def _check_max_heap_consistency(h: MaxHeap):
    keys = h._keys
    values = h._values
    indexes = h._indexes

    #print(keys, values, indexes)

    assert len(keys) == len(values) == len(indexes)

    for i,v in enumerate(values):
        if 2*i + 1 < len(values):
            assert v >= values[2*i + 1], \
                f"{v} = heap[{i}] <= heap[{2*i + 1}] = {values[2*i + 1]}, {values}"
        if 2*i + 2 < len(values):
            assert v >= values[2*i + 2], \
                f"{v} = heap[{i}] <= heap[{2*i + 2}] = {values[2*i + 2]}, {values}"

    indexes_keys = set(h._indexes.keys())
    expected_indexes_keys = set(keys)
    assert indexes_keys == expected_indexes_keys, \
        f"{indexes_keys} != {expected_indexes_keys}"

    indexes_positions = set(h._indexes.values())
    expected_indexes_positions = set(range(len(keys)))
    assert indexes_positions == expected_indexes_positions, \
        f"{indexes_positions} != {expected_indexes_positions}"

    print('Max heap: OK!')


if __name__ == '__main__':
    l = [0,2,1,4,3,6,5,8,7,9]

    h = MinHeap(keys=range(len(l)), values=l)
    _check_min_heap_consistency(h)

    h.update(0, -1)
    _check_min_heap_consistency(h)

    h.update(5, 100)
    _check_min_heap_consistency(h)

    h.pop()
    _check_min_heap_consistency(h)

    h.delete(2)
    _check_min_heap_consistency(h)

    popped = h.replace(-5, 100)
    _check_min_heap_consistency(h)


    h = MaxHeap(keys=range(len(l)), values=l)
    _check_max_heap_consistency(h)

    h.update(0, -1)
    _check_max_heap_consistency(h)

    h.update(5, 100)
    _check_max_heap_consistency(h)

    h.pop()
    _check_max_heap_consistency(h)

    h.delete(2)
    _check_max_heap_consistency(h)

    popped = h.replace(-5, 100)
    _check_max_heap_consistency(h)
