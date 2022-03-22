import os

from collections import Counter
from itertools import combinations, product, repeat

from utils.datamodels import InstanceData


def check_phase1_state(data: InstanceData, cover_idxs, cover_cost, current_costs,
        element_counts, incident_subsets, unique_elements, check_cover = False):

    errors = []

    expected_current_costs = data.subset_costs.copy()
    for pair in product(range(data.n_subsets), cover_idxs):
        expected_current_costs[pair[0]] += data.conflict_costs.get(pair, 0)
    for i, (c1, c2) in enumerate(zip(current_costs, expected_current_costs)):
        if c1 != c2:
            errors.append(f"current_costs[{i}] = {c1} != {c2} = expected_cc[{i}]")

    expected_cover_cost = (
        sum(expected_current_costs[idx] for idx in cover_idxs) +
        - sum(data.conflict_costs.get(p, 0) for p in combinations(cover_idxs, 2))
    )
    if expected_cover_cost != cover_cost:
        errors.append(f"cover_cost = {cover_cost} != expected_cover_cost = {expected_cover_cost}")

    l = len(element_counts)
    if l != data.n_elements:
        errors.append(f"element_counts has length {l} != {data.n_elements}")

    ec_counter = Counter(e for idx in cover_idxs for e in data.subsets[idx])
    expected_ec = [ec_counter[e_idx] for e_idx in range(data.n_elements)]
    for i, (c1, c2) in enumerate(zip(element_counts, expected_ec)):
        if c1 != c2:
            errors.append(f"element_counts[{i}] = {c1} != {c2} = expected_ec[{i}]")

    expected_incident_subsets = [set() for _ in repeat(None, data.n_elements)]
    for idx in cover_idxs:
        for e in data.subsets[idx]:
            expected_incident_subsets[e].add(idx)
    if len(expected_incident_subsets) != len(incident_subsets):
        errors.append(
            f"len(expected_incident_subsets) = {len(expected_incident_subsets)} "
            f"!= {len(incident_subsets)} = len(incident_subsets)"
        )
    for e, exp_subs in enumerate(expected_incident_subsets):
        if incident_subsets[e] != exp_subs:
            errors.append(
                f"expected_incident_subsets[{e}] = {expected_incident_subsets[e]} != "
                f"{incident_subsets[e]} = incident_subsets[{e}]"
            )

    expected_unique_elements = [set() for _ in repeat(None, data.n_subsets)]
    for e, subs in enumerate(expected_incident_subsets):
        if len(subs) == 1:
            expected_unique_elements[next(iter(subs))].add(e)
    if len(expected_unique_elements) != len(unique_elements):
        errors.append(
            f"len(expected_unique_elements) = {len(expected_unique_elements)} "
            f"!= {len(unique_elements)} = len(unique_elements)"
        )
    for idx, exp_elms in enumerate(expected_unique_elements):
        if unique_elements[idx] != exp_elms:
            errors.append(
                f"expected_unique_elements[{idx}] = {expected_unique_elements[idx]} != "
                f"{unique_elements[idx]} = unique_elements[{idx}]"
            )

    if check_cover:
        actual_cover = set().union(*(data.subsets[idx] for idx in cover_idxs))
        expected_cover = set(range(data.n_elements))
        if actual_cover != expected_cover:
            errors.append(
                f"Missing elements inside cover: {expected_cover.difference(actual_cover)}, "
                f"exceeding elements inside cover: {actual_cover.difference(expected_cover)}"
            )

    error_count = len(errors)
    if error_count > 0:
        pid = os.getpid()
        assert False, (
            f"Worker {pid}, {error_count} error(s) detected (cover: {cover_idxs}):"
            f'\n  - Worker {pid}: ' + f'\n  - Worker {pid}: '.join(errors)
        )


def check_phase2_state(data: InstanceData, cover_idxs: set[int],
        cover_cost: int, current_costs: list[int], element_counts: list[int],
        covering_subsets: list[set[int]], unique_elements: list[set[int]],
        availables: set[int]):

    errors = []

    actual_union = cover_idxs.union(availables)
    expected_union = set(range(data.n_subsets))
    if actual_union != expected_union:
        errors.append(
            f"Missing elements: {expected_union.difference(actual_union)}, "
            f"exceeding elements: {actual_union.difference(expected_union)}"
        )

    actual_cover = set().union(*(data.subsets[idx] for idx in cover_idxs))
    expected_cover = set(range(data.n_elements))
    if actual_cover != expected_cover:
        errors.append(
            f"Missing elements inside cover: {expected_cover.difference(actual_cover)}, "
            f"exceeding elements inside cover: {actual_cover.difference(expected_cover)}"
        )

    overlap = cover_idxs.intersection(availables)
    if overlap:
        errors.append(
            f"Intersection between cover and availables is not empty: {overlap}"
        )

    expected_current_costs = data.subset_costs.copy()
    for pair in product(range(data.n_subsets), cover_idxs):
        expected_current_costs[pair[0]] += data.conflict_costs.get(pair, 0)
    for i, (c1, c2) in enumerate(zip(current_costs, expected_current_costs)):
        if c1 != c2:
            errors.append(
                f"current_costs[{i}] = {c1}  !=  {c2} = expected_current_cost[{i}]\n"
            )

    expected_cover_cost = (
        sum(expected_current_costs[idx] for idx in cover_idxs) -
        sum(data.conflict_costs.get(p, 0) for p in combinations(cover_idxs, 2))
    )
    if expected_cover_cost != cover_cost:
        errors.append(f"cover_cost = {cover_cost} != expected_cover_cost = {expected_cover_cost}")

    ec_counter = Counter(e for idx in cover_idxs for e in data.subsets[idx])
    expected_element_counts = [ec_counter.get(e_idx, -1) for e_idx in range(data.n_elements)]
    for e, (c1, c2) in enumerate(zip(element_counts, expected_element_counts)):
        if c1 != c2:
            errors.append(f"element_counts[{e}] = {c1}  !=  {c2} = expected_element_counts[{e}]")

    expected_covering_subsets = [set() for e in range(data.n_elements)]
    for idx in cover_idxs:
        for e in data.subsets[idx]:
            expected_covering_subsets[e].add(idx)
    if expected_covering_subsets != covering_subsets:
        for e, (exp, act) in enumerate(zip(expected_covering_subsets, covering_subsets)):
            if exp != act:
                errors.append(
                    f"expected_covering_subsets[{e}] = {exp} != {act} = covering_subsets[{e}]"
                )

    expected_unique_elements = [set() for _ in repeat(None, data.n_subsets)]
    for e, subs in enumerate(expected_covering_subsets):
        if len(subs) == 1:
            expected_unique_elements[next(iter(subs))].add(e)
    if len(expected_unique_elements) != len(unique_elements):
        errors.append(
            f"len(expected_unique_elements) = {len(expected_unique_elements)} "
            f"!= {len(unique_elements)} = len(unique_elements)"
        )
    for idx, exp_elms in enumerate(expected_unique_elements):
        if unique_elements[idx] != exp_elms:
            errors.append(
                f"expected_unique_elements[{idx}] = {expected_unique_elements[idx]} != "
                f"{unique_elements[idx]} = unique_elements[{idx}]"
            )

    error_count = len(errors)
    if error_count > 0:
        pid = os.getpid()
        assert False, (
            f"Worker {pid}, {error_count} error(s) detected (cover: {cover_idxs}):"
            f'\n  - Worker {pid}: ' + f'\n  - Worker {pid}: '.join(errors)
        )
