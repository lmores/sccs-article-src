import csv
import os
from traceback import format_exc

from utils.config import PROJECT_ROOT
from utils.datamodels import (
    AnalysisResult,
    BaseResult,
    GraspResult,
    GurobiResult,
    SccsBaseResult,
)
from utils.vprint import v0_print

from .misc import to_human_datetime


# CSV headers sequences
_SCCS_RESULT_CSV_HEADERS = (
    # Main info
    'INSTANCE_ID', 'ACTION', 'STATUS', 'CMD_ARGS', 'CPU_COUNT', 'TIME_LIMIT',

    # Instance info
    'N_ELEMENTS', 'N_SUBSETS', 'CONFLICT_THRESHOLD',
    'SUBSETS_COST', 'NON_ZERO_SUBSET_COSTS_COUNT',
    'SUBSET_COSTS_DISTRIB', 'AGGREGATED_SUBSET_COSTS_DISTRIB',
    'CONFLICTS_COST', 'NON_ZERO_CONFLICT_COSTS_COUNT',
    'CONFLICT_COSTS_DISTRIB', 'AGGREGATED_CONFLICT_COSTS_DISTRIB',
    'MAX_CONFLICTS',

    # Runtime info
    'START_AT', 'END_AT', 'USER_TIME', 'SYSTEM_TIME',
    'READ_TIME', 'INIT_TIME', 'TIME_TO_BEST', 'SOLVE_TIME', 'TOTAL_TIME',

    # Solution info
    'SOLUTION', 'SOLUTION_COST', 'SOLUTION_VALID',
    'SOLUTION_SUBSETS_COST', 'SOLUTION_NON_ZERO_SUBSET_COSTS_COUNT',
    'SOLUTION_SUBSET_COSTS_DISTRIB', 'SOLUTION_AGGREGATED_SUBSET_COSTS_DISTRIB',
    'SOLUTION_CONFLICTS_COST', 'SOLUTION_NON_ZERO_CONFLICT_COSTS_COUNT',
    'SOLUTION_CONFLICT_COSTS_DISTRIB', 'SOLUTION_AGGREGATED_CONFLICT_COSTS_DISTRIB',

    # Other info
    'LOG_FILE_PATH', 'RESULTS_FILE_PATH', 'ANALYSES_FILE_PATH',
    'PYTHON_INTERPRETER', 'GIT_HASH', 'RUNTIME_EXCEPTION'
)

_GRASP_SCCS_RESULT_CSV_HEADERS = _SCCS_RESULT_CSV_HEADERS + (
    # Main info
    'WORKER_COUNT',

    # Parameters info
    'STAGES',
    'MAX_CANDIDATES',
    'ITERATIONS_LIMIT',
    'STALE_ITERATIONS_LIMIT',

    'S1_MAX_CANDIDATES',
    'S1_TIME_LIMIT',
    'S1_ITERATIONS_LIMIT',
    'S1_STALE_ITERATIONS_LIMIT',

    'S2_MAX_CANDIDATES',
    'S2_TIME_LIMIT',
    'S2_ITERATIONS_LIMIT',
    'S2_STALE_ITERATIONS_LIMIT',

    # Runtime info
    'INTER_STAGE_TIME',

    # Execution info
    'ITERATIONS_COUNT',
    'P1_SHARED_CACHE_HIT_COUNT',
    'P1_SHARED_CACHE_MISS_COUNT',
    'P2_H0_MOVE_COUNT',
    'P2_H1_MOVE_COUNT',
    'P2_1H_MOVE_COUNT',

    # Stage 1 summary
    'S1_STATUS',
    'S1_SOLUTION',
    'S1_SOLUTION_COST',
    'S1_TIME_TO_BEST',

    # Stage 2 summary
    'S2_STATUS',
    'S2_SOLUTION',
    'S2_SOLUTION_COST',
    'S2_TIME_TO_BEST',
)

_GUROBI_SCCS_RESULT_CSV_HEADERS = _SCCS_RESULT_CSV_HEADERS + (
    # Gurobi info
    'RELAXED',
    'SOLUTION_BEST_BOUND',
    'SOLUTION_GAP',
    'GUROBI_VERSION'
)

_ANALYSIS_CSV_HEADERS = (
    # Main info
    'INSTANCE_ID', 'ACTION', 'STATUS', 'CMD_ARGS', 'CPU_COUNT', 'TIME_LIMIT',

    # Instance info
    'N_ELEMENTS', 'N_SUBSETS', 'CONFLICT_THRESHOLD',
    'SUBSETS_COST', 'NON_ZERO_SUBSET_COSTS_COUNT',
    'SUBSET_COSTS_DISTRIB', 'AGGREGATED_SUBSET_COSTS_DISTRIB',
    'CONFLICTS_COST', 'NON_ZERO_CONFLICT_COSTS_COUNT',
    'CONFLICT_COSTS_DISTRIB', 'AGGREGATED_CONFLICT_COSTS_DISTRIB',
    'MAX_CONFLICTS',

    # Runtime info
    'START_AT', 'END_AT',
    'USER_TIME', 'SYSTEM_TIME',
    'READ_TIME', 'TOTAL_TIME',

    # Analysis related info
    'ROW_SUMS', 'ROW_AVG', 'ROW_STDDEV',
    'COLUMN_SUMS', 'COLUMN_AVG', 'COLUMN_STDDEV',
    #'UNIQUE_ELEMENTS', 'CONNECTED_COMPONENTS_COUNT', 'CONNECTED_COMPONENT_SIZES',
    'NON_EMPTY_INTERSECTIONS_COUNT', 'INTERSECTION_SIZES_DISTRIB',
    'AGGREGATED_INTERSECTION_SIZES_DISTRIB',

    # Runtime related info
    'LOG_FILE_PATH', 'RESULTS_FILE_PATH', 'ANALYSES_FILE_PATH',
    'PYTHON_INTERPRETER', 'GIT_HASH', 'RUNTIME_EXCEPTION'
)


def save_result(result: BaseResult) -> bool:
    v0_print("[INFO] Saving result of instance '{}' (k={})",
        result.instance_id, result.conflict_threshold)

    try:
        if isinstance(result, SccsBaseResult):
            file_path = result.results_file_path
            if not os.path.exists(file_path):
                os.makedirs(file_path.parent, exist_ok=True)
                with open(file_path, 'w', newline='') as fp:
                    csv_writer = csv.DictWriter(fp, _SCCS_RESULT_CSV_HEADERS)
                    csv_writer.writeheader()

            base_result_dict = {
                # Main info
                'INSTANCE_ID': result.instance_id,
                'ACTION': result.action.name,
                'STATUS': result.status.name,
                'CMD_ARGS': ' '.join(result.cmd_args),
                'CPU_COUNT': result.cpu_count,
                'TIME_LIMIT': result.time_limit,

                # Instance info
                'N_ELEMENTS': result.n_elements,
                'N_SUBSETS': result.n_subsets,
                'CONFLICT_THRESHOLD': result.conflict_threshold,

                'SUBSETS_COST': result.subsets_cost,
                'NON_ZERO_SUBSET_COSTS_COUNT': result.non_zero_subset_costs_count,
                'SUBSET_COSTS_DISTRIB': result.subset_costs_distrib,
                'AGGREGATED_SUBSET_COSTS_DISTRIB': result.aggregated_subset_costs_distrib,

                'CONFLICTS_COST': result.conflicts_cost,
                'NON_ZERO_CONFLICT_COSTS_COUNT': result.non_zero_conflict_costs_count,
                'CONFLICT_COSTS_DISTRIB': result.conflict_costs_distrib,
                'AGGREGATED_CONFLICT_COSTS_DISTRIB': result.aggregated_conflict_costs_distrib,
                'MAX_CONFLICTS': result.max_conflicts,

                # Runtime info
                'START_AT': to_human_datetime(result.start_at),
                'END_AT': to_human_datetime(result.end_at),
                'USER_TIME': result.user_time,
                'SYSTEM_TIME': result.system_time,
                'READ_TIME': result.read_time,
                'INIT_TIME': result.init_time,
                'TIME_TO_BEST': result.time_to_best,
                'SOLVE_TIME': result.solve_time,
                'TOTAL_TIME': result.total_time,

                # Solution related info
                'SOLUTION': result.solution,
                'SOLUTION_COST': result.solution_cost,
                'SOLUTION_VALID': result.solution_valid,

                'SOLUTION_SUBSETS_COST': result.solution_subsets_cost,
                'SOLUTION_NON_ZERO_SUBSET_COSTS_COUNT': result.solution_non_zero_subset_costs_count,
                'SOLUTION_SUBSET_COSTS_DISTRIB': result.solution_subset_costs_distrib,
                'SOLUTION_AGGREGATED_SUBSET_COSTS_DISTRIB': result.solution_aggregated_subset_costs_distrib,

                'SOLUTION_CONFLICTS_COST': result.solution_conflicts_cost,
                'SOLUTION_NON_ZERO_CONFLICT_COSTS_COUNT': result.solution_non_zero_conflict_costs_count,
                'SOLUTION_CONFLICT_COSTS_DISTRIB': result.solution_conflict_costs_distrib,
                'SOLUTION_AGGREGATED_CONFLICT_COSTS_DISTRIB': result.solution_aggregated_conflict_costs_distrib,

                # Runtime info
                'LOG_FILE_PATH': result.log_file_path.relative_to(PROJECT_ROOT),
                'RESULTS_FILE_PATH': result.results_file_path.relative_to(PROJECT_ROOT),
                'ANALYSES_FILE_PATH': result.analyses_file_path.relative_to(PROJECT_ROOT),
                'PYTHON_INTERPRETER': result.python_interpreter,
                'GIT_HASH': result.git_hash,
                'RUNTIME_EXCEPTION': result.runtime_exception
            }

            with open(file_path, 'a', newline='') as fp:
                csv_writer = csv.DictWriter(fp, _SCCS_RESULT_CSV_HEADERS)
                csv_writer.writerow(base_result_dict)

            if isinstance(result, GraspResult):
                grasp_file_path = file_path.with_stem(f'{file_path.stem}_grasp')
                if not os.path.exists(grasp_file_path):
                    with open(grasp_file_path, 'w', newline='') as fp:
                        csv_writer = csv.DictWriter(fp, _GRASP_SCCS_RESULT_CSV_HEADERS)
                        csv_writer.writeheader()

                grasp_result_dict = base_result_dict.copy()
                grasp_result_dict.update({
                    # Main info
                    'WORKER_COUNT': result.worker_count,

                    # Parameters info
                    'STAGES': result.stages,
                    'MAX_CANDIDATES': result.max_candidates,
                    'ITERATIONS_LIMIT': result.iterations_limit,
                    'STALE_ITERATIONS_LIMIT': result.stale_iterations_limit,

                    'S1_MAX_CANDIDATES': result.s1_max_candidates,
                    'S1_TIME_LIMIT': result.s1_time_limit,
                    'S1_ITERATIONS_LIMIT': result.s1_iterations_limit,
                    'S1_STALE_ITERATIONS_LIMIT': result.s1_stale_iterations_limit,

                    'S2_MAX_CANDIDATES': result.s1_max_candidates,
                    'S2_TIME_LIMIT': result.s1_time_limit,
                    'S2_ITERATIONS_LIMIT': result.s1_iterations_limit,
                    'S2_STALE_ITERATIONS_LIMIT': result.s1_stale_iterations_limit,

                    # Runtime info
                    'INTER_STAGE_TIME': result.inter_stage_time,

                    # Execution info
                    'ITERATIONS_COUNT': result.iterations_count,
                    'P1_SHARED_CACHE_HIT_COUNT': result.p1_shared_cache_hit_count,
                    'P1_SHARED_CACHE_MISS_COUNT': result.p1_shared_cache_miss_count,
                    'P2_H0_MOVE_COUNT': result.p2_h0_move_count,
                    'P2_H1_MOVE_COUNT': result.p2_h1_move_count,
                    'P2_1H_MOVE_COUNT': result.p2_1h_move_count,

                    # Stage 1 summary
                    'S1_STATUS': result.s1_status.name,
                    'S1_SOLUTION': result.s1_solution,
                    'S1_SOLUTION_COST': result.s1_solution_cost,
                    'S1_TIME_TO_BEST': result.s1_time_to_best,

                    # Stage 2 summary
                    'S2_STATUS': result.s2_status.name,
                    'S2_SOLUTION': result.s2_solution or None,
                    'S2_SOLUTION_COST': result.s2_solution_cost,
                    'S2_TIME_TO_BEST': result.s2_time_to_best,
                })

                with open(grasp_file_path, 'a', newline='') as fp:
                    csv_writer = csv.DictWriter(fp, _GRASP_SCCS_RESULT_CSV_HEADERS)
                    csv_writer.writerow(grasp_result_dict)

            elif isinstance(result, GurobiResult):
                gurobi_file_path = file_path.with_stem(f'{file_path.stem}_gurobi')
                if not os.path.exists(gurobi_file_path):
                    with open(gurobi_file_path, 'w', newline='') as fp:
                        csv_writer = csv.DictWriter(fp, _GUROBI_SCCS_RESULT_CSV_HEADERS)
                        csv_writer.writeheader()

                gurobi_result_dict = base_result_dict.copy()
                gurobi_result_dict.update({
                    # Gurobi info
                    'RELAXED': result.relaxed,
                    'SOLUTION_BEST_BOUND': result.solution_best_bound,
                    'SOLUTION_GAP': result.solution_gap,
                    'GUROBI_VERSION': result.gurobi_version
                })

                with open(gurobi_file_path, 'a', newline='') as fp:
                    csv_writer = csv.DictWriter(fp, _GUROBI_SCCS_RESULT_CSV_HEADERS)
                    csv_writer.writerow(gurobi_result_dict)

        elif isinstance(result, AnalysisResult):
            if not os.path.exists(result.analyses_file_path):
                os.makedirs(result.analyses_file_path.parent, exist_ok=True)
                with open(result.analyses_file_path, 'w', newline='') as fp:
                    csv_writer = csv.DictWriter(fp, _ANALYSIS_CSV_HEADERS)
                    csv_writer.writeheader()

            with open(result.analyses_file_path, 'a', newline='') as fp:
                csv_writer = csv.DictWriter(fp, _ANALYSIS_CSV_HEADERS)
                csv_writer.writerow({
                    # Main info
                    'INSTANCE_ID': result.instance_id,
                    'ACTION': result.action.name,
                    'STATUS': result.status.name,
                    'CMD_ARGS': ' '.join(result.cmd_args),
                    'CPU_COUNT': result.cpu_count,
                    'TIME_LIMIT': result.time_limit,

                    # Instance info
                    'N_ELEMENTS': result.n_elements,
                    'N_SUBSETS': result.n_subsets,
                    'CONFLICT_THRESHOLD': result.conflict_threshold,

                    'SUBSETS_COST': result.subsets_cost,
                    'NON_ZERO_SUBSET_COSTS_COUNT': result.non_zero_subset_costs_count,
                    'SUBSET_COSTS_DISTRIB': result.subset_costs_distrib,
                    'AGGREGATED_SUBSET_COSTS_DISTRIB': result.aggregated_subset_costs_distrib,

                    'CONFLICTS_COST': result.conflicts_cost,
                    'NON_ZERO_CONFLICT_COSTS_COUNT': result.non_zero_conflict_costs_count,
                    'CONFLICT_COSTS_DISTRIB': result.conflict_costs_distrib,
                    'AGGREGATED_CONFLICT_COSTS_DISTRIB': result.aggregated_conflict_costs_distrib,
                    'MAX_CONFLICTS': result.max_conflicts,

                    # Runtime info
                    'START_AT': to_human_datetime(result.start_at),
                    'END_AT': to_human_datetime(result.end_at),
                    'USER_TIME': result.user_time,
                    'SYSTEM_TIME': result.system_time,
                    'READ_TIME': result.read_time,
                    'TOTAL_TIME': result.total_time,

                    # Analysis info
                    'ROW_SUMS': result.row_sums,
                    'ROW_AVG': result.row_avg,
                    'ROW_STDDEV': result.row_stddev,
                    'COLUMN_SUMS': result.column_sums,
                    'COLUMN_AVG': result.column_avg,
                    'COLUMN_STDDEV': result.column_stddev,
                    #'UNIQUE_ELEMENTS': result.unique_elements,
                    #'CONNECTED_COMPONENTS_COUNT': result.connected_components_count,
                    #'CONNECTED_COMPONENT_SIZES': result.connected_component_sizes,
                    'NON_EMPTY_INTERSECTIONS_COUNT': result.non_empty_intersections_count,
                    'INTERSECTION_SIZES_DISTRIB': result.intersection_sizes_distrib,
                    'AGGREGATED_INTERSECTION_SIZES_DISTRIB': result.aggregated_intersection_sizes_distrib,

                    # Runtime related info
                    'LOG_FILE_PATH': result.log_file_path.relative_to(PROJECT_ROOT),
                    'RESULTS_FILE_PATH': result.results_file_path.relative_to(PROJECT_ROOT),
                    'ANALYSES_FILE_PATH': result.analyses_file_path.relative_to(PROJECT_ROOT),
                    'PYTHON_INTERPRETER': result.python_interpreter,
                    'GIT_HASH': result.git_hash,
                    'RUNTIME_EXCEPTION': result.runtime_exception
                })

    except:
        v0_print("[ERROR] The following exception occurred while saving "
            "this result:\n{}\n{}", result, format_exc())
        return False

    return True
