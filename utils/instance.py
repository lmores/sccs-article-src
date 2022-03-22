import math
import os

from itertools import combinations
from pathlib import Path
from typing import Iterable

from utils.datamodels import InstanceData
from utils.config import DATASETS_DIR_PATH, FLOAT_ABS_TOLERANCE


# Dataset readers
def read_instance(instance_id: str, conflict_threshold: int) -> InstanceData:
    # The 'conflict_threshold' argument is passed in as it is not written in
    # the data set file and its needed to compute the conflict costs

    file_format = instance_id.split(os.path.sep, maxsplit=1)[0]
    if "beasley" == file_format:
        return _read_beasley_instance(instance_id, conflict_threshold)
    else:
        raise SyntaxError(
            f"Could not read data set file of type '{file_format}' "
            f"for instance '{instance_id}' (k={conflict_threshold})"
        )


def _read_beasley_instance(instance_id: str, conflict_threshold: int) -> InstanceData:
    file_path = Path(DATASETS_DIR_PATH, instance_id)

    with file_path.open('r') as fp:
        n_elements, n_subsets = map(int, fp.readline().split())

        subset_costs = []
        while len(subset_costs) < n_subsets:
            subset_costs.extend(map(int, fp.readline().split()))
        assert len(subset_costs) == n_subsets

        subsets = [set() for _ in range(n_subsets)]
        for e in range(n_elements):
            _count = 0
            _repetitions, *one_based_idxs = map(int, fp.readline().split())
            for one_based_idx in one_based_idxs:
                subsets[one_based_idx - 1].add(e)
                _count += 1
            while _count < _repetitions:
                one_based_idxs = map(int, fp.readline().split())
                for one_based_idx in one_based_idx:
                    subsets[one_based_idx - 1].add(e)
                    _count += 1
            assert _count == _repetitions

    conflict_costs = {}
    coeff = round(max(c / len(s) for c, s in zip(subset_costs, subsets))) or 1
    n_subsets = len(subsets)
    for i in range(n_subsets - 1):
        for j in range(i + 1, n_subsets):
            intersection_size = len(subsets[i].intersection(subsets[j]))
            conflict_size = max(intersection_size - conflict_threshold, 0)
            if conflict_size != 0:
                conflict_cost = coeff * conflict_size
                conflict_costs[i,j] = conflict_cost
                conflict_costs[j,i] = conflict_cost

    return InstanceData(id=instance_id, n_elements=n_elements,
        n_subsets=n_subsets, subsets=subsets, subset_costs=subset_costs,
        conflict_costs=conflict_costs, conflict_threshold=conflict_threshold)


# Check solution
def check_sccs_solution(solution: Iterable[int], solution_cost: float,
        data: InstanceData) -> bool | str:

    actual_elements = set().union(*(data.subsets[j] for j in solution))
    if len(actual_elements) != data.n_elements:
        return (
            f"Found {len(actual_elements)} elements in solution, "
            f"but {data.n_elements} were expected, missing elements: "
            f"{set(range(data.n_elements)).difference(actual_elements)}"
        )

    expected_cost = (
        sum(data.subset_costs[i] for i in solution) +
        sum(data.conflict_costs.get(p, 0) for p in combinations(solution, 2))
    )

    return (math.isclose(expected_cost, solution_cost, abs_tol=FLOAT_ABS_TOLERANCE) or
        f"Solution {solution}: cost is {solution_cost}, but {expected_cost} "
        f"was expected, {list((i, data.subset_costs[i]) for i in solution)} "
        f"{list((p, data.conflict_costs.get(p, 0)) for p in combinations(solution, 2))}")
