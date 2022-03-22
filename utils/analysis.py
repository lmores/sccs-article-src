import os
import statistics
import sys

from collections import Counter
from itertools import combinations
from pathlib import Path
from psutil import Process
from time import time
from typing import Iterable

from utils.cmd_args import Action
from utils.config import LOGS_DIR_PATH, LOG_FILE_NAME_TEMPLATE, PROJECT_ROOT
from utils.datamodels import InstanceData, AnalysisResult, ResultStatus, SolutionAnalysisResult
from utils.instance import check_sccs_solution
from utils.misc import aggregate_distrib, build_distrib, current_human_datetime, to_human_datetime
#from utils.unionfind import UnionFind
from utils.vprint import v0_print


# Instance analysis
def analyse_instance(instance: InstanceData) -> AnalysisResult:
    log_file_path = Path(LOGS_DIR_PATH, LOG_FILE_NAME_TEMPLATE.format(
        base=os.path.splitext(instance.id)[0],
        type='analysis',
        datetime=current_human_datetime()
    ))

    result = AnalysisResult(instance.id, Action.ANALYSE,
        cmd_args=sys.argv, cpu_count=1, log_file_path=log_file_path)
    result.add_instance_data(instance)

    process = Process()
    cpu_times_start = process.cpu_times()
    result.start_at = time()

    row_sums = [0 for _ in range(instance.n_elements)]
    result.row_sums = row_sums
    for subset in instance.subsets:
        for e in subset:
            row_sums[e] += 1
    result.column_sums = [len(s) for s in instance.subsets]
    result.row_avg = statistics.fmean(result.row_sums)
    result.row_stddev = statistics.pstdev(result.row_sums)
    result.column_avg = statistics.fmean(result.column_sums)
    result.column_stddev = statistics.pstdev(result.column_sums)

    intersection_sizes = tuple(len(s.intersection(t)) for s,t in combinations(instance.subsets, 2))
    result.intersection_sizes_distrib = Counter(intersection_sizes)
    result.aggregated_intersection_sizes_distrib = aggregate_distrib(result.intersection_sizes_distrib)
    result.non_empty_intersections_count = (
        len(intersection_sizes) - result.intersection_sizes_distrib.get(0, 0)
    )

    # result.connected_component_sizes = []
    # components = UnionFind(data.n_elements)
    # for j in range(data.n_subsets):
    #     representative = None
    #     for i in range(data.n_elements):
    #         if i in data.subsets[j]:
    #             if representative is None:
    #                 representative = i
    #             else:
    #                 components.union(representative, i)
    # result.connected_component_sizes = [components.sizes[x] for x in components.roots]
    # result.connected_components_count = len(result.connected_component_sizes)
    #result.unique_elements = [e for e,s in enumerate(row_sums) if s == 1]

    result.end_at = time()
    cpu_times_end = process.cpu_times()
    result.user_time = cpu_times_end.user - cpu_times_start.user
    result.system_time = cpu_times_end.system - cpu_times_start.system
    result.total_time = result.end_at - result.start_at
    result.status = ResultStatus.DONE

    save_analysis_result(result)

    return result


def build_analysis_info(result: AnalysisResult) -> str:
    return (
        f">>>>> '{result.instance_id}' ANALYSIS RESULT <<<<<\n"
        f"Elements: {result.n_elements}\n"
        f"Subsets: {result.n_subsets}\n"
        f"Conflict threshold: {result.conflict_threshold}\n"

        f"Subsets cost: {result.subsets_cost}\n"
        f"Non zero subset costs count: {result.non_zero_subset_costs_count} "
            f"({result.non_zero_subset_costs_count / result.n_subsets:%})\n"
        f"Subset costs distrib: {result.subset_costs_distrib}\n"
        f"Aggregated subset costs distrib: {result.aggregated_subset_costs_distrib}\n"

        f"Conflicts cost: {result.conflicts_cost}\n"
        f"Non zero conflict costs count: {result.non_zero_conflict_costs_count} "
            f"({result.non_zero_conflict_costs_count / result.max_conflicts:.2%})\n"
        f"Conflict costs distrib: {result.conflict_costs_distrib}\n"
        f"Aggregated conflict costs distrib: {result.aggregated_conflict_costs_distrib}\n"
        f"Max conflicts: {result.max_conflicts}\n"

        f"Row sums: {result.row_sums}\n"
        f"Row average ± stddev: {result.row_avg} ± {result.row_stddev}\n"
        f"Column sums: {result.column_sums}\n"
        f"Column average ± stddev: {result.column_avg} ± {result.column_stddev}\n"
        #f"Unique elements: {result.unique_elements}\n"
        #f"Connected components count: {result.connected_components_count}\n"
        #f"Connected component sizes: {result.connected_component_sizes}\n"
        f"Non empty intersections count: {result.non_empty_intersections_count} "
            f"({result.non_empty_intersections_count / result.max_conflicts:%})\n"
        f"Intersection sizes distrib: {result.intersection_sizes_distrib}\n"
        f"Aggregated intersection sizes distrib: {result.aggregated_intersection_sizes_distrib}\n"

        f"\n>>>>> '{result.instance_id}' EXECUTION INFO <<<<<\n"
        f"Action: {result.action}\n"
        f"Status: {result.status.name}\n"
        f"Cmd args: {' '.join(result.cmd_args)}\n"
        f"Cpu count: {result.cpu_count}\n"
        f"Time limit: {result.time_limit}{' seconds' if result.time_limit is not None else ''}\n"

        f"\n>>>>> '{result.instance_id}' RUNTIME INFO <<<<<\n"
        f"Start at: {to_human_datetime(result.start_at)}\n"
        f"End at: {to_human_datetime(result.end_at)}\n"
        f"User time: {result.user_time}\n"
        f"System time: {result.system_time}\n"
        #f"Read time: {result.read_time}\n"
        f"Total time: {result.total_time}\n"
        # Note: 'read_time' is not available at this point as it is set in main.py

        f"\n>>>>> '{result.instance_id}' OTHER INFO <<<<<\n"
        f"Log file path: {result.log_file_path.relative_to(PROJECT_ROOT)}\n"
        f"Python interpreter: {result.python_interpreter}\n"
        f"Git hash: {result.git_hash}\n"
        f"Runtime exception: {result.runtime_exception}\n"
        # Note: 'result.results_file_path' and 'result.analyses_file_path' are
        # not available at this point as they are set in main.py
    )


def save_analysis_result(result: AnalysisResult) -> None:
    v0_print("[INFO] Writing analysis result file: {}", result.log_file_path)
    os.makedirs(result.log_file_path.parent, exist_ok=True)

    # Write log file
    with open(result.log_file_path, 'w') as fp:
        fp.write(build_analysis_info(result))

    # Write json file
    json_file_path = result.log_file_path.with_suffix('.json')
    with open(json_file_path, 'w') as fp:
        fp.write(result.to_json())


# Solution analysis
def analyse_solution(solution: Iterable[int], solution_cost: float,
        data: InstanceData) -> SolutionAnalysisResult:

    result = SolutionAnalysisResult()
    result.valid = check_sccs_solution(solution, solution_cost, data)

    # Subset costs distrib
    solution_subset_costs = tuple(data.subset_costs[idx] for idx in solution)
    result.subsets_cost = sum(solution_subset_costs)
    result.subset_costs_distrib = build_distrib(solution_subset_costs)
    result.aggregated_subset_costs_distrib = aggregate_distrib(result.subset_costs_distrib)
    result.non_zero_subset_costs_count = (
        len(solution_subset_costs) - result.subset_costs_distrib.get(0, 0)
    )

    # Conflict costs distrib
    solution_conflict_costs = tuple(data.conflict_costs.get((i,j), 0)
                                        for i,j in combinations(solution, 2))
    result.conflicts_cost = sum(solution_conflict_costs)
    result.conflict_costs_distrib = build_distrib(solution_conflict_costs)
    result.aggregated_conflict_costs_distrib = aggregate_distrib(result.conflict_costs_distrib)
    result.non_zero_conflict_costs_count = (
        len(solution_conflict_costs) - result.conflict_costs_distrib.get(0, 0)
    )

    return result
