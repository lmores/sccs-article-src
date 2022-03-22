class UnionFind():
    def __init__(self, n_elements: int):
        if isinstance(n_elements, int):
            self.n_elements = n_elements
            self.parents: dict[int,int] = {k:k for k in range(n_elements)}
            self.sizes: dict[int,int] = {k:1 for k in range(n_elements)}  # number of descendants (including the element itself)
            self.roots: set = set(range(n_elements))
        else:
            raise TypeError(
                f"Argument 'n_elements' is not an integer, "
                f"got '{n_elements}' ({n_elements.__class__})")

    def find_root(self, x):
        while self.parents[x] != x:
            x, self.parents[x] = self.parents[x], self.parents[self.parents[x]]  # Path halving

        try:
            pass #assert x in self.roots
        except AssertionError as e:
            print(self.parents)
            print(self.roots)
            print(set(self.parents.keys()).difference(self.roots))
            print(set(self.roots).difference(self.parents.keys()))
            raise e

        return x

    def union(self, x, y):
        #if x == y:  return

        xroot = self.find_root(x)
        yroot = self.find_root(y)

        if xroot == yroot:
            return

        # Union by size
        if self.sizes[xroot] > self.sizes[yroot]:
            xroot, yroot = yroot, xroot

        self.parents[xroot] = yroot
        self.sizes[yroot] += self.sizes[xroot]

        del self.sizes[xroot]
        self.roots.discard(xroot)

    def is_root(self, x):
        return self.parents[x] == x

    def number_of_components(self):
        assert len(self.roots) == sum(1 for x in self.parents if self.is_root(x))
        return len(self.roots)

    def __str__(self):
        return f"UnionType object with parents: {self.parents} and sizes: {self.sizes}"


def main():
    import collections
    import itertools
    import os
    from .instance import read_instance
    from .misc import nat_sort_key

    for path, _, file_names in os.walk('datasets/beasley/new'):
        file_names.sort(key=nat_sort_key)

        for file_name in file_names:
            instance_id = os.path.join(path, file_name).removeprefix('datasets' + os.path.sep)
            n_elements, n_subsets, incidence_matrix, _ = read_instance(instance_id)
            uf = UnionFind(n_elements)
            c = collections.Counter(i for i,j in itertools.product(range(n_elements), range(n_subsets)) if incidence_matrix[i][j] == 1)
            #print()
            #print(sorted(c.items(), key=lambda x: x[1])[:100])
            #continue
            for j in range(n_subsets):
                repr_element = None
                for i in range(n_elements):
                    if incidence_matrix[i][j] == 1:
                        if repr_element is None:
                            repr_element = i
                        else:
                            #print(repr_element, i)
                            uf.union(repr_element, i)

            print(
                f"{file_name} - components: {uf.number_of_components()} {[uf.sizes[x] for x in uf.roots]}, "
                f"unique elements: {[k for k,v in c.items() if v == 1]}"
            )

if __name__ == '__main__':
    main()
