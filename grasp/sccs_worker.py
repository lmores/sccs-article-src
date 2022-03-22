import os

from collections import Counter
from cProfile import Profile
from itertools import combinations, product, repeat
from marshal import dumps, loads
from math import ceil, sqrt
from multiprocessing import Queue, Value
from pathlib import Path
from pstats import Stats
from psutil import Process
from typing import NamedTuple
from time import time

from utils.datamodels import GraspStageResult, GraspWorkerResult, InstanceData, ResultStatus
from utils.heaps import MaxHeap, MinHeap
from utils.vprint import set_verbosity_level, v0_print, v1_print, v2_print

from .debug import check_phase1_state, check_phase2_state
from .utils import build_worker_result_short_info


class InstanceInfo(NamedTuple):
    feasible: bool
    initial_cover: set[int]
    initial_cover_cost: int
    incident_subsets: list[set[int]]
    conflicting_subsets: list[set[int]]

class MoveItem(NamedTuple):
    stack_idx: int
    subset_idxs: frozenset[int]
    cost_delta: int
    current_costs: dict[int,int]
    element_counts: dict[int,int]

class SharedState(NamedTuple):
    working_cover_cost: int
    current_costs: list[int]
    element_counts: list[int]
    max_heap: MaxHeap
    min_heap: MinHeap


def run_one_stage_sccs(
        data: InstanceData,
        max_candidates: int | None,
        shared_cache: dict[frozenset[int],SharedState],
        incumbent: Value,
        selected_idxs: set[int] | None = None,
        time_limit: float | None = None,
        iterations_limit: int | None = None,
        stale_iterations_limit: int | None = None,
        verbosity: int=0,
        silent: bool=False,
        log_file_path: Path | None = None,
        profile: bool=False
    ) -> GraspWorkerResult:

    # Setup
    set_verbosity_level(level=verbosity, silent=silent)

    pid = os.getpid()
    worker_log_file_path = Path(
        log_file_path.parent,
        log_file_path.stem,    # name of the directory for worker log files
        f"{log_file_path.stem}_{pid}.log"
    )

    # Start profiler
    if profile:
        profiler = Profile()
        profiler.enable()
    else:
        profiler = None

    # Init
    init_time_start = time()
    info = build_instance_info(data)
    init_time = time() - init_time_start

    # Solve
    stage1_result = _solve_sccs(data, info, max_candidates, shared_cache,
        incumbent, selected_idxs, time_limit, iterations_limit, stale_iterations_limit)
    stage1_result.stage = 1

    # Stop profiler and save stats to file
    if profile:
        profiler.disable()
        stats_file_path = worker_log_file_path.with_suffix('.stats')
        os.makedirs(stats_file_path.parent, exist_ok=True)
        with open(stats_file_path, 'w') as fp:
            stats = Stats(profiler, stream=fp)
            stats.dump_stats(stats_file_path)

    # Build result
    result = GraspWorkerResult(instance_id=data.id, pid=pid, stages=1,
        log_file_path=worker_log_file_path,
        init_time=init_time,
        s1_max_candidates=max_candidates,
        s1_time_limit=time_limit,
        s1_iterations_limit=iterations_limit,
        s1_stale_iterations_limit=stale_iterations_limit,
        s2_max_candidates=max_candidates, s2_time_limit=time_limit,
        s2_iterations_limit=iterations_limit,
        s2_stale_iterations_limit=stale_iterations_limit)
    result.add_first_stage_info(stage1_result)

    # Display worker solution info
    if verbosity > 0 and not silent:
        msg = build_worker_result_short_info(result)
        v1_print(msg, flush=True)

    return result


def run_two_stage_sccs(
        data: InstanceData,
        max_candidates: int | None,
        shared_cache: dict[frozenset[int],SharedState],
        incumbent: Value,
        outbound_queue: Queue,
        inbound_queue: Queue,
        stage1_time_limit: float | None = None,
        stage2_time_limit: float | None = None,
        iterations_limit: int | None = None,
        stale_iterations_limit: int | None = None,
        verbosity: int=0,
        silent: bool=False,
        log_file_path: Path | None = None,
        profile: bool=False
    ) -> GraspWorkerResult:

    # Setup
    set_verbosity_level(level=verbosity, silent=silent)

    pid = os.getpid()
    log_file_path = Path(
        log_file_path.parent,
        log_file_path.stem,    # name of the directory for worker log files
        f"{log_file_path.stem}_{pid}.log"
    )

    # Start profiler
    if profile:
        profiler = Profile()
        profiler.enable()
    else:
        profiler = None

    # Solve
    solve_time_start = time()

    # Init
    init_time_start = time()
    info = build_instance_info(data)
    init_time = time() - init_time_start

    # Execute first stage
    stage1_result = _solve_sccs(data, info, max_candidates, shared_cache,
        incumbent, None, stage1_time_limit, iterations_limit, stale_iterations_limit)
    stage1_result.stage = 1

    # Send the result of the first stage to the main process
    inter_stage_time_start = time()
    outbound_queue.put(stage1_result, block=False)

    # Wait for the set of subset indexes to use in phase 1 from the main proces
    stage2_idxs = inbound_queue.get()
    inter_stage_time = time() - inter_stage_time_start

    # Execute second stage
    stage2_time_start = time()
    stage2_result = _solve_sccs(data, info, max_candidates, shared_cache,
        incumbent, stage2_idxs, stage2_time_limit, iterations_limit, stale_iterations_limit)
    stage2_result.stage = 2

    solve_time = time() - solve_time_start

    # Stop profiler and save stats to file
    if profile:
        profiler.disable()
        stats_file_path = log_file_path.with_suffix('.stats')
        os.makedirs(stats_file_path.parent, exist_ok=True)
        with open(stats_file_path, 'w') as fp:
            stats = Stats(profiler, stream=fp)
            stats.dump_stats(stats_file_path)

    # Build result
    result = GraspWorkerResult(instance_id=data.id, pid=pid, stages=2,
        log_file_path=log_file_path,
        init_time=init_time,
        inter_stage_time = inter_stage_time,
        solve_time = solve_time,
        s1_max_candidates=max_candidates,
        s1_time_limit=stage1_time_limit,
        s1_iterations_limit=iterations_limit,
        s1_stale_iterations_limit=stale_iterations_limit,
        s2_max_candidates=max_candidates,
        s2_time_limit=stage2_time_limit,
        s2_iterations_limit=iterations_limit,
        s2_stale_iterations_limit=stale_iterations_limit
    )
    result.add_first_stage_info(stage1_result)
    result.add_second_stage_info(stage2_result, stage2_time_start - solve_time_start)

    # Send result to main process
    outbound_queue.put(result)


def build_instance_info(data: InstanceData) -> InstanceInfo:
    initial_cover = set(range(data.n_subsets))
    initial_cover_cost = 0

    subset_costs = data.subset_costs
    incident_subsets = [set() for _ in repeat(None, data.n_elements)]
    for idx, subset in enumerate(data.subsets):
        initial_cover_cost += subset_costs[idx]
        for e in subset:
            incident_subsets[e].add(idx)
    feasible = all(incident_subsets)

    conflicting_subsets = [set() for _ in repeat(None, data.n_subsets)]
    for (idx, jdx), cost in data.conflict_costs.items():
        if idx < jdx:
            initial_cover_cost += cost
            conflicting_subsets[idx].add(jdx)
            conflicting_subsets[jdx].add(idx)

    return InstanceInfo(feasible, initial_cover, initial_cover_cost,
        incident_subsets, conflicting_subsets)


def _solve_sccs(
        data: InstanceData,
        info: InstanceInfo,
        max_candidates: int | None,
        shared_cache: dict[frozenset[int],SharedState],
        incumbent: Value,
        p1_selected_idxs: set[int] | None = None,
        time_limit: float | None = None,
        iterations_limit: int | None = None,
        stale_iterations_limit: int | None = None,
    ) -> GraspStageResult:

    # Setup
    pid = os.getpid()

    # Read-only data structures
    instance_id = data.id
    n_elements = data.n_elements
    n_subsets = data.n_subsets
    subsets = data.subsets
    subset_costs = data.subset_costs
    conflict_costs = data.conflict_costs
    conflict_threshold = data.conflict_threshold
    incident_subsets = info.incident_subsets
    conflicting_subsets = info.conflicting_subsets

    if max_candidates is None:
        max_candidates = ceil(sqrt(n_subsets))

    if p1_selected_idxs is None:
        p1_selected_idxs = list(range(n_subsets))
    p1_selected_idxs_count = len(p1_selected_idxs)

    result = GraspStageResult(instance_id=data.id, pid=pid,
        p1_selected_idxs=p1_selected_idxs,
        p1_selected_idxs_count=p1_selected_idxs_count)

    v0_print(
        "[INFO] Worker {}: solving SCCS instance '{}' using GRASP "
        "(conflict_threshold={}, max_candidates={}, phase1_selected_subsets_count={} "
        "time_limit={}, iterations_limit={}, stale_iterations_limit={})",
        pid, instance_id, conflict_threshold, max_candidates, p1_selected_idxs_count,
        time_limit, iterations_limit, stale_iterations_limit, flush=True
    )

    # Data shared across iterations
    best_cover_idxs = info.initial_cover
    best_cover_cost = info.initial_cover_cost
    topk_size = min(max_candidates, p1_selected_idxs_count)

    # Data structures reused during each iteration of phase 1
    __delete_ws_idxs = []
    __redundant_wc_idxs = set()

    # Data structures reused during each iteration of phase 2
    __free_idxs = []
    __incompatible_idxs = []
    __current_costs = {}
    __ec_changes = {}
    __element_candidates = {}
    __skip = set()
    __outgoing_candidate_idxs: list[int] = []
    __EMPTY_CURRENT_COSTS = {}
    __EMPTY_ELEMENT_COUNTS = {}
    __EMPTY_REMOVED = frozenset()

    # Result info
    time_to_best = 0
    p1_shared_cache_hit_count = 0
    p1_shared_cache_miss_count = 0
    p2_h0_move_count = 0
    p2_h1_move_count = 0
    p2_1h_move_count = 0
    result__history = result.history
    result__p1_durations = result.p1_durations
    result__p1_final_costs = result.p1_final_costs
    result__p1_repetitions = result.p1_repetitions
    result__p2_durations = result.p2_durations
    result__p2_final_costs = result.p2_final_costs
    result__p2_improvements = result.p2_improvements
    result__p2_repetitions = result.p2_repetitions

    # Solve
    process = Process()
    iterations_count, stale_iterations_count = 0, 0
    solve_time_start, cpu_times_start = time(), process.cpu_times()
    while (
            (time_limit is None or time() - solve_time_start < time_limit) and
            (iterations_limit is None or iterations_count < iterations_limit) and
            (stale_iterations_limit is None or stale_iterations_count < stale_iterations_limit)
        ):

        # Data for current iteration
        working_cover_cost = 0
        working_cover_idxs = set()
        working_subsets = {idx:subsets[idx].copy() for idx in p1_selected_idxs}
        current_costs = subset_costs.copy()
        element_counts = [0 for _ in repeat(None, n_elements)]
        unique_elements = [set() for _ in repeat(None, n_subsets)]
        covering_subsets = [set() for _ in repeat(None, n_elements)]

        # Phase I: build a "good" feasible solution
        p1_time_start = time()

        min_heap = MinHeap((idx, subset_costs[idx] / len(subsets[idx])) for idx in p1_selected_idxs)
        max_heap = MaxHeap(min_heap.pop() for _ in repeat(None, topk_size))

        ### DEBUG
        # check_phase1_state(data, working_cover_idxs, working_cover_cost,
        #     current_costs, element_counts, covering_subsets, unique_elements)

        p1_repetitions = 0
        while working_subsets:
            p1_repetitions += 1

            # Randomly select an element to build a feasible solution
            chosen_idx, _ = max_heap.pop_random()
            working_cover_idxs.add(chosen_idx)
            frozen_working_cover_idxs = frozenset(working_cover_idxs)
            cached_state = shared_cache.get(frozen_working_cover_idxs, None)
            if cached_state is None:
                p1_shared_cache_miss_count += 1

                del working_subsets[chosen_idx]
                chosen_subset = subsets[chosen_idx]
                working_cover_cost += current_costs[chosen_idx]
                if min_heap:
                    max_heap.push(*min_heap.pop())

                __redundant_wc_idxs.clear()
                for e in chosen_subset:
                    new_count = element_counts[e] = element_counts[e] + 1
                    if new_count == 1:
                        unique_elements[chosen_idx].add(e)
                    elif new_count == 2:
                        other_idx = next(iter(covering_subsets[e]))
                        unique_elements[other_idx].remove(e)
                        if not unique_elements[other_idx]:
                            __redundant_wc_idxs.add(other_idx)
                    covering_subsets[e].add(chosen_idx)
                for idx in conflicting_subsets[chosen_idx]:
                    current_costs[idx] += conflict_costs[idx, chosen_idx]
                    if idx in min_heap:
                        min_heap.update(idx, current_costs[idx] / len(working_subsets[idx]))
                    elif idx in max_heap:
                        max_heap.update(idx, current_costs[idx] / len(working_subsets[idx]))

                # Remove redundant subsets and update the cost
                for idx in __redundant_wc_idxs:
                    if unique_elements[idx]:
                        continue
                    working_cover_idxs.remove(idx)
                    unique_elements[idx].clear()
                    working_cover_cost -= current_costs[idx]
                    for e in subsets[idx]:
                        element_counts[e] -= 1
                        covering_subsets[e].remove(idx)
                        if element_counts[e] == 1:
                            other_idx = next(iter(covering_subsets[e]))
                            unique_elements[other_idx].add(e)
                    for jdx in conflicting_subsets[idx]:
                        current_costs[jdx] -= conflict_costs[idx, jdx]
                        if jdx in min_heap:
                            min_heap.update(jdx, current_costs[jdx] / len(working_subsets[jdx]))
                        elif jdx in max_heap:
                            max_heap.update(jdx, current_costs[jdx] / len(working_subsets[jdx]))

                # Restore heaps ordering
                while min_heap and max_heap.peek_value() > min_heap.peek_value():
                    min_item = min_heap.peek()
                    max_item = max_heap.replace(*min_item)
                    min_heap.replace(*max_item)

                # Update working subsets
                __delete_ws_idxs.clear()
                for idx, subset in working_subsets.items():
                    subset -= chosen_subset
                    if not subset:
                        __delete_ws_idxs.append(idx)
                for idx in __delete_ws_idxs:
                    del working_subsets[idx]
                    if min_heap.discard(idx) is None:
                        max_heap.delete(idx)
                        if min_heap:
                            max_heap.push(*min_heap.pop())

                ### DEBUG
                # check_phase1_state(data, working_cover_idxs, working_cover_cost,
                #     current_costs, element_counts, covering_subsets, unique_elements)

                shared_cache[frozenset(working_cover_idxs)] = dumps((
                    working_subsets, working_cover_cost, current_costs,
                    element_counts, covering_subsets, unique_elements,
                    max_heap._keys, max_heap._values, max_heap._indexes,
                    min_heap._keys, min_heap._values, min_heap._indexes))

            else:
                p1_shared_cache_hit_count += 1

                (working_subsets, working_cover_cost, current_costs,
                    element_counts, covering_subsets, unique_elements,
                    maxh_keys, maxh_values, maxh_indexes,
                    minh_keys, minh_values, minh_indexes) = loads(cached_state)
                max_heap.set_state(maxh_keys, maxh_values, maxh_indexes)
                min_heap.set_state(minh_keys, minh_values, minh_indexes)

                ### DEBUG
                # check_phase1_state(data, working_cover_idxs, working_cover_cost,
                #     current_costs, element_counts, covering_subsets, unique_elements)

        ### DEBUG
        # check_phase1_state(data, working_cover_idxs, working_cover_cost,
        #     current_costs, element_counts, covering_subsets, unique_elements, True)

        # Store result info
        result__p1_durations.append(time() - p1_time_start)
        result__p1_final_costs.append(working_cover_cost)
        result__p1_repetitions.append(p1_repetitions)

        # Phase II: local search
        p2_start_time = time()
        p2_repetitions = 0

        availables = set(idx for idx in range(n_subsets))
        availables.difference_update(working_cover_idxs)

        ### DEBUG
        # check_phase2_state(data, working_cover_idxs, working_cover_cost,
        #    current_costs, element_counts, covering_subsets, unique_elements, availables)

        while True:
            # (h,0)-move, h > 0 (elimination)
            cost_delta = 0
            outgoing_idxs: frozenset[int] | None = None
            changed_element_counts: dict[int, int] | None = None
            removable_idxs = []
            removable_ec = Counter[int]()
            for idx in working_cover_idxs:
                if not unique_elements[idx]:
                    removable_idxs.append(idx)
                    removable_ec.update(subsets[idx])
            if removable_idxs:
                critical_elements = [e for e,rc in removable_ec.items() if rc == element_counts[e]]
                __free_idxs.clear()
                __incompatible_idxs.clear()
                for idx in removable_idxs:
                    if any(e in subsets[idx] for e in critical_elements):
                        __incompatible_idxs.append(idx)
                    else:
                        __free_idxs.append(idx)

                for idx in __free_idxs:
                    availables.add(idx)
                    working_cover_idxs.remove(idx)
                    working_cover_cost -= current_costs[idx]
                    for jdx in conflicting_subsets[idx]:
                        current_costs[jdx] -= conflict_costs[idx, jdx]
                    for e in subsets[idx]:
                        covering_subsets[e].remove(idx)
                        element_counts[e] -= 1
                        if element_counts[e] == 1:
                            covering_idx = next(iter(covering_subsets[e]))
                            unique_elements[covering_idx].add(e)

                ii_size = len(__incompatible_idxs)
                if ii_size > 0:
                    __skip.clear()
                    stack = [
                        MoveItem(i, __EMPTY_REMOVED, 0, __EMPTY_CURRENT_COSTS,
                            __EMPTY_ELEMENT_COUNTS) for i in range(ii_size)
                    ]
                    while stack:
                        (_i, _removed_idxs, _cost_delta, _current_costs, _element_counts) = stack.pop()
                        idx = __incompatible_idxs[_i]
                        _removed_idxs = _removed_idxs.union((idx,))
                        if _removed_idxs in __skip:
                            continue

                        __ec_changes.clear()
                        for e in subsets[idx]:
                            count = _element_counts.get(e, element_counts[e])
                            if count < 2:
                                __skip.add(_removed_idxs)
                                break
                            __ec_changes[e] = count - 1
                        else:
                            _cost_delta -= _current_costs.get(idx, current_costs[idx])
                            _element_counts = _element_counts.copy()
                            _element_counts.update(__ec_changes)
                            _current_costs = _current_costs.copy()
                            for jdx in conflicting_subsets[idx]:
                                cc = _current_costs.get(jdx, current_costs[jdx])
                                _current_costs[jdx] = cc - conflict_costs[idx, jdx]

                            if len(_removed_idxs) > 1:
                                for jdx in _removed_idxs:
                                    __skip.add(_removed_idxs.difference((jdx,)))

                            for _j in range(_i + 1, ii_size):
                                stack.append(MoveItem(_j, _removed_idxs,
                                    _cost_delta, _current_costs, _element_counts))

                            if _cost_delta < cost_delta:
                                cost_delta = _cost_delta
                                outgoing_idxs = _removed_idxs
                                changed_element_counts = _element_counts

                    if cost_delta < 0:
                        working_cover_cost += cost_delta
                        for o_idx in outgoing_idxs:
                            availables.add(o_idx)
                            working_cover_idxs.remove(o_idx)
                            for idx in conflicting_subsets[o_idx]:
                                current_costs[idx] -= conflict_costs[idx, o_idx]
                            for e in subsets[o_idx]:
                                covering_subsets[e].remove(o_idx)

                        for e, e_count in changed_element_counts.items():
                            if e_count == 1:
                                assert len(covering_subsets[e]) == 1, covering_subsets[e]
                                covering_idx = next(iter(covering_subsets[e]))
                                unique_elements[covering_idx].add(e)
                            element_counts[e] = e_count

                p2_repetitions += 1
                p2_h0_move_count += 1

                ### DEBUG
                # check_phase2_state(data, working_cover_idxs, working_cover_cost,
                #     current_costs, element_counts, covering_subsets, unique_elements, availables)

                continue

            # (h,1)-move, h > 0
            cost_delta = 0
            incoming_idx: int | None = None
            outgoing_idxs: frozenset[int] | None = None
            changed_element_counts = None
            for candidate_incoming_idx in availables:
                candidate_incoming_subset = subsets[candidate_incoming_idx]

                __outgoing_candidate_idxs.clear()
                for idx in working_cover_idxs:
                    if unique_elements[idx].issubset(candidate_incoming_subset):
                        __outgoing_candidate_idxs.append(idx)

                if not __outgoing_candidate_idxs:
                    continue

                _element_counts = {e: element_counts[e]+1 for e in candidate_incoming_subset}
                _current_costs = {}
                for idx in conflicting_subsets[candidate_incoming_idx]:
                    _current_costs[idx] = (
                        current_costs[idx] +
                        conflict_costs[idx, candidate_incoming_idx]
                    )

                __skip.clear()
                oci_size = len(__outgoing_candidate_idxs)
                stack = [MoveItem(oci_idx, __EMPTY_REMOVED, current_costs[candidate_incoming_idx],
                    _current_costs, _element_counts) for oci_idx in range(oci_size)]

                while stack:
                    (_idx, _removed_idxs, _cost_delta, _current_costs, _element_counts) = stack.pop()
                    new_outgoing_idx = __outgoing_candidate_idxs[_idx]
                    _removed_idxs = _removed_idxs.union((new_outgoing_idx,))
                    if _removed_idxs in __skip:
                        continue

                    __ec_changes.clear()
                    for e in subsets[new_outgoing_idx]:
                        count = _element_counts.get(e, element_counts[e])
                        if count < 2:
                            __skip.add(_removed_idxs)
                            break
                        __ec_changes[e] = count - 1
                    else:
                        _cost_delta -= _current_costs.get(new_outgoing_idx, current_costs[new_outgoing_idx])
                        _element_counts = _element_counts.copy()
                        _element_counts.update(__ec_changes)
                        _current_costs = _current_costs.copy()
                        for idx in conflicting_subsets[new_outgoing_idx]:
                            cc = _current_costs.get(idx, current_costs[idx])
                            _current_costs[idx] = cc - conflict_costs[idx, new_outgoing_idx]

                        if len(_removed_idxs) > 1:
                            for idx in _removed_idxs:
                                __skip.add(_removed_idxs.difference((idx,)))

                        for _jdx in range(_idx + 1, oci_size):
                            stack.append(MoveItem(_jdx, _removed_idxs,
                                _cost_delta, _current_costs, _element_counts))

                        if _cost_delta < cost_delta:
                            cost_delta = _cost_delta
                            outgoing_idxs = _removed_idxs
                            incoming_idx = candidate_incoming_idx
                            changed_element_counts = _element_counts

            if cost_delta < 0:
                # Apply best (h,1)-move found
                working_cover_cost += cost_delta

                for o_idx in outgoing_idxs:
                    availables.add(o_idx)
                    working_cover_idxs.remove(o_idx)
                    unique_elements[o_idx].clear()
                    for idx in conflicting_subsets[o_idx]:
                        current_costs[idx] -= conflict_costs[idx, o_idx]
                    for e in subsets[o_idx]:
                        covering_subsets[e].remove(o_idx)

                availables.remove(incoming_idx)
                working_cover_idxs.add(incoming_idx)
                for idx in conflicting_subsets[incoming_idx]:
                    current_costs[idx] += conflict_costs[idx, incoming_idx]
                for e in subsets[incoming_idx]:
                    covering_subsets[e].add(incoming_idx)

                for e, e_count in changed_element_counts.items():
                    if e_count == 1:
                        assert len(covering_subsets[e]) == 1, covering_subsets[e]
                        covering_idx = next(iter(covering_subsets[e]))
                        unique_elements[covering_idx].add(e)
                    elif element_counts[e] == 1:
                        for s_idx in covering_subsets[e]:
                            unique_elements[s_idx].discard(e)  #TODO: check discard()
                    element_counts[e] = e_count

                p2_repetitions += 1
                p2_h1_move_count += 1

                ### DEBUG
                # check_phase2_state(data, working_cover_idxs, working_cover_cost,
                #     current_costs, element_counts, covering_subsets, unique_elements, availables)

                continue

            # (1,h)-move, h > 1
            cost_delta = 0
            outgoing_idx: int
            incoming_idxs: frozenset[int]
            outgoing_candidate_idxs = sorted(working_cover_idxs, key=lambda x: current_costs[x])
            for candidate_outgoing_idx in outgoing_candidate_idxs:
                _base_cost_delta = -current_costs[candidate_outgoing_idx]
                if _base_cost_delta > cost_delta:
                    break

                __current_costs.clear()
                for idx in conflicting_subsets[candidate_outgoing_idx]:
                    __current_costs[idx] = (
                        current_costs[idx] -
                        conflict_costs[idx, candidate_outgoing_idx]
                    )

                __element_candidates.clear()
                for e in unique_elements[candidate_outgoing_idx]:
                    __element_candidates[e] = sorted(incident_subsets[e],
                        key=lambda x: __current_costs.get(x, current_costs[x]))

                swap_move_start = time()
                local_coverings = map(frozenset, product(*__element_candidates.values()))
                for local_cover_idxs in local_coverings:
                    _outer_conflict_delta = sum(
                        __current_costs.get(idx, current_costs[idx])
                            for idx in local_cover_idxs
                    )
                    _cost_delta = _base_cost_delta + _outer_conflict_delta
                    if _cost_delta < cost_delta:
                        _cost_delta += sum(
                            conflict_costs.get(p, 0)
                                for p in combinations(local_cover_idxs, 2)
                        )
                        if _cost_delta < cost_delta:
                            cost_delta = _cost_delta
                            incoming_idxs = local_cover_idxs
                            outgoing_idx = candidate_outgoing_idx
                    if time() - swap_move_start > .01:  break

            if cost_delta < 0:
                # Apply best (1,h)-move found
                working_cover_cost += cost_delta

                availables.add(outgoing_idx)
                working_cover_idxs.remove(outgoing_idx)
                unique_elements[outgoing_idx].clear()
                for idx in conflicting_subsets[outgoing_idx]:
                    current_costs[idx] -= conflict_costs[idx, outgoing_idx]
                for e in subsets[outgoing_idx]:
                    covering_subsets[e].remove(outgoing_idx)

                for i_idx in incoming_idxs:
                    availables.remove(i_idx)
                    working_cover_idxs.add(i_idx)
                    for idx in conflicting_subsets[i_idx]:
                        current_costs[idx] += conflict_costs[idx, i_idx]
                    for e in subsets[i_idx]:
                        covering_subsets[e].add(i_idx)

                element_count_delta = Counter()
                element_count_delta.subtract(subsets[outgoing_idx])
                for idx in incoming_idxs:
                    element_count_delta.update(subsets[idx])
                for e, delta in element_count_delta.items():
                    old_count = element_counts[e]
                    new_count = old_count + delta
                    element_counts[e] = new_count
                    if new_count == 1:
                        covering_idx = next(iter(covering_subsets[e]))
                        unique_elements[covering_idx].add(e)
                    elif old_count == 1:
                        for s_idx in covering_subsets[e]:
                            unique_elements[s_idx].discard(e)  #TODO: check discard()

                p2_repetitions += 1
                p2_1h_move_count += 1

                ### DEBUG
                # check_phase2_state(data, working_cover_idxs, working_cover_cost,
                #     current_costs, element_counts, covering_subsets, unique_elements, availables)

                continue

            p2_repetitions += 1
            break

        ### DEBUG
        # check_phase2_state(data, working_cover_idxs, working_cover_cost,
        #     current_costs, element_counts, covering_subsets, unique_elements, availables)

        # Store result info
        result__p2_durations.append(time() - p2_start_time)
        result__p2_final_costs.append(working_cover_cost)
        result__p2_improvements.append(result__p1_final_costs[-1] - working_cover_cost)
        result__p2_repetitions.append(p2_repetitions)

        # Check if a better solution has been found
        if working_cover_cost < best_cover_cost:
            now = time()
            time_to_best = now - solve_time_start
            best_cover_idxs = working_cover_idxs
            best_cover_cost = working_cover_cost

            with incumbent.get_lock():
                if working_cover_cost < incumbent.value:
                    incumbent.value = working_cover_cost
                    stale_iterations_count = 0
                else:
                    stale_iterations_count += 1

            improvement_scope = 'LOCAL' if stale_iterations_count else 'GLOBAL'
            event_msg = (
                f"[INFO] Worker {pid}: {improvement_scope} solution improved. "
                f"Cost {best_cover_cost}, solution: {best_cover_idxs}"
            )
            result__history.append((now, event_msg))
            v2_print(event_msg)
        else:
            stale_iterations_count += 1
        iterations_count += 1

    # Save result info
    cpu_times_end = process.cpu_times()
    result.user_time = cpu_times_end.user - cpu_times_start.user
    result.system_time = cpu_times_end.system - cpu_times_start.system
    result.solve_time = time() - solve_time_start
    result.time_to_best = time_to_best
    result.solution = best_cover_idxs
    result.solution_cost = best_cover_cost
    result.iterations_count = iterations_count
    result.stale_iterations_count = stale_iterations_count
    result.p1_shared_cache_hit_count = p1_shared_cache_hit_count
    result.p1_shared_cache_miss_count = p1_shared_cache_miss_count
    result.p2_h0_move_count = p2_h0_move_count
    result.p2_h1_move_count = p2_h1_move_count
    result.p2_1h_move_count = p2_1h_move_count

    if time_limit and result.solve_time > time_limit:
        result.status = ResultStatus.TIME_LIMIT
    else:
        result.status = ResultStatus.DONE

    return result
