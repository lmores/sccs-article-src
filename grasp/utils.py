import os

from copy import deepcopy
from pathlib import Path
from pstats import Stats

from utils.config import PROJECT_ROOT
from utils.datamodels import GraspResult, GraspStageResult, GraspWorkerResult
from utils.misc import to_human_datetime


class MockSharedDict(dict):
    _SENTINEL = object()

    def __setitem__(self, key, value):
        super().__setitem__(key, deepcopy(value))

    def __getitem__(self, key):
        return deepcopy(super().__getitem__(key))

    def get(self, key, fallback=_SENTINEL):
        if fallback is MockSharedDict._SENTINEL:
            return deepcopy(super().get(key))

        return deepcopy(super().get(key, fallback))


def build_result_info(result: GraspResult) -> str:
    return (
        f">>>>> '{result.instance_id}' SOLUTION INFO <<<<<\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Solution valid: {result.solution_valid}\n"
        f"Solution subsets cost: {result.solution_subsets_cost}\n"
        f"Solution non zero subset costs count: {result.solution_non_zero_subset_costs_count}\n"
        f"Solution subset costs distrib: {result.solution_subset_costs_distrib}\n"
        f"Solution aggregated subset costs distrib: {result.solution_aggregated_subset_costs_distrib}\n"
        f"Solution conflicts cost: {result.solution_conflicts_cost}\n"
        f"Solution non zero conflict costs count: {result.solution_non_zero_conflict_costs_count}\n"
        f"Solution conflict costs distrib: {result.solution_conflict_costs_distrib}\n"
        f"Solution aggregated conflict costs distrib: {result.solution_aggregated_conflict_costs_distrib}\n"

        f"\n>>>>> '{result.instance_id}' EXECUTION INFO <<<<<\n"
        f"Action: {result.action}\n"
        f"Status: {result.status.name}\n"
        f"Cmd args: {' '.join(result.cmd_args)}\n"
        f"Cpu count: {result.cpu_count}\n"
        f"Workers count: {result.worker_count}\n"

        f"\n>>>>> '{result.instance_id}' PARAMETERS INFO <<<<<\n"
        f"Max candidates: {result.max_candidates}\n"
        f"Time limit: {result.time_limit}{' seconds' if result.time_limit is not None else ''}\n"
        f"Iterations limit: {result.iterations_limit}\n"
        f"Stale iterations limit: {result.stale_iterations_limit}\n"

        f"\n>>>>> '{result.instance_id}' RUNTIME INFO <<<<<\n"
        f"Start at: {to_human_datetime(result.start_at)}\n"
        f"End at: {to_human_datetime(result.end_at)}\n"
        f"User time: {result.user_time}\n"
        f"System time: {result.system_time}\n"
        f"Read time: {result.read_time}\n"
        f"Init time: {result.init_time}\n"
        f"Inter stage time: {result.inter_stage_time}\n"
        f"Time to best: {result.time_to_best}\n"
        f"Solve time: {result.solve_time}\n"
        f"Total time: {result.total_time}\n"

        f"\n>>>>> '{result.instance_id}' STAGE 1 SUMMARY <<<<<\n"
        f"Status: {result.s1_status.name}\n"
        f"Solution: {result.s1_solution}\n"
        f"Solution cost: {result.s1_solution_cost}\n"
        f"Time to best: {result.s1_time_to_best}\n"

        f"\n>>>>> '{result.instance_id}' STAGE 2 SUMMARY <<<<<\n"
        f"Status: {result.s2_status.name}\n"
        f"Solution: {result.s2_solution or None}\n"
        f"Solution cost: {result.s2_solution_cost}\n"
        f"Time to best: {result.s2_time_to_best}\n"

        f"\n>>>>> '{result.instance_id}' INSTANCE INFO <<<<<\n"
        f"Elements: {result.n_elements}\n"
        f"Subsets: {result.n_subsets}\n"
        f"Conflict threshold: {result.conflict_threshold}\n"

        f"Subsets cost: {result.subsets_cost}\n"
        f"Non zero subset costs count: {result.non_zero_subset_costs_count}\n"
        f"Subset costs distrib: {result.subset_costs_distrib}\n"
        f"Aggregated subset costs distrib: {result.aggregated_subset_costs_distrib}\n"

        f"Conflicts cost: {result.conflicts_cost}\n"
        f"Non zero conflict costs count: {result.non_zero_conflict_costs_count}\n"
        f"Conflict costs distrib: {result.conflict_costs_distrib}\n"
        f"Aggregated conflict costs distrib: {result.aggregated_conflict_costs_distrib}\n"
        f"Max conflicts: {result.max_conflicts}\n"

        f"\n>>>>> '{result.instance_id}' OTHER INFO <<<<<\n"
        f"Log file path: {result.log_file_path.relative_to(PROJECT_ROOT)}\n"
        f"Python interpreter: {result.python_interpreter}\n"
        f"Git hash: {result.git_hash}\n"
        f"Runtime exception: {result.runtime_exception}\n"
        # Note: 'result.results_file_path' and 'result.analyses_file_path' are
        # not available as this point as they are set in main.py
    )

def build_result_short_info(result: GraspResult) -> str:
    return (
        f">>>>> '{result.instance_id}' (k={result.conflict_threshold}) GRASP RESULT <<<<<\n"
        f"Status: {result.status.name}\n"
        f"Start at: {to_human_datetime(result.start_at)}\n"
        f"End at: {to_human_datetime(result.end_at)}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Init time: {result.init_time} seconds\n"
        f"Inter stage time: {result.inter_stage_time} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Iterations count: {result.iterations_count}\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Solution valid: {result.solution_valid}\n"
        f"Solution subsets cost: {result.solution_subsets_cost}\n"
        f"Solution non zero subset costs count: {result.solution_non_zero_subset_costs_count}\n"
        f"Solution subset costs distrib: {result.solution_subset_costs_distrib}\n"
        f"Solution aggregated subset costs distrib: {result.solution_aggregated_subset_costs_distrib}\n"
        f"Solution conflicts cost: {result.solution_conflicts_cost}\n"
        f"Solution non zero conflict costs count: {result.solution_non_zero_conflict_costs_count}\n"
        f"Solution conflict costs distrib: {result.solution_conflict_costs_distrib}\n"
        f"Solution aggregated conflict costs distrib: {result.solution_aggregated_conflict_costs_distrib}\n"
    )


def build_worker_result_info(result: GraspWorkerResult) -> str:
    return (
        f">>> '{result.instance_id}' WORKER RESULT (pid: {result.pid}) <<<\n"
        f"Stages: {result.stages}\n"
        f"Status: {result.status.name}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Init time: {result.init_time} seconds\n"
        f"Inter stage time: {result.inter_stage_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Iterations count: {result.iterations_count}\n"
        f"Phase 1 shared cache hit count: {result.p1_shared_cache_hit_count}\n"
        f"Phase 1 shared cache miss count: {result.p1_shared_cache_miss_count}\n"
        f"Phase 2 (h,0) moves count: {result.p2_h0_move_count}\n"
        f"Phase 2 (h,1) moves count: {result.p2_h1_move_count}\n"
        f"Phase 2 (1,h) moves count: {result.p2_1h_move_count}\n"

        f"Stage 1 status: {result.s1_status.name}\n"
        f"Stage 1 user time: {result.s1_user_time} seconds\n"
        f"Stage 1 system time: {result.s1_system_time} seconds\n"
        f"Stage 1 solve time: {result.s1_solve_time} seconds\n"
        f"Stage 1 time to best: {result.s1_time_to_best} seconds\n"
        f"Stage 1 sterations count: {result.s1_iterations_count}\n"
        f"Stage 1 stale iterations count: {result.s1_stale_iterations_count}\n"
        f"Stage 1 solution: {result.s1_solution}\n"
        f"Stage 1 solution cost: {result.s1_solution_cost}\n"
        f"Stage 1 phase 1 selected subsets: {result.s1_p1_selected_idxs}\n"
        f"Stage 1 phase 1 selected subsets count: {result.s1_p1_selected_idxs_count}\n"
        f"Stage 1 phase 1 shared cache hit count: {result.s1_p1_shared_cache_hit_count}\n"
        f"Stage 1 phase 1 shared cache miss count: {result.s1_p1_shared_cache_miss_count}\n"
        f"Stage 1 phase 1 repetitions: {result.s1_p1_repetitions}\n"
        f"Stage 1 phase 1 durations: {result.s1_p1_durations}\n"
        f"Stage 1 phase 1 final costs: {result.s1_p1_final_costs}\n"
        f"Stage 1 phase 2 (h,0) moves count: {result.s1_p2_h0_move_count}\n"
        f"Stage 1 phase 2 (h,1) moves count: {result.s1_p2_h1_move_count}\n"
        f"Stage 1 phase 2 (1,h) moves count: {result.s1_p2_1h_move_count}\n"
        f"Stage 1 phase 2 repetitions: {result.s1_p2_repetitions}\n"
        f"Stage 1 phase 2 improvements: {result.s1_p2_improvements}\n"
        f"Stage 1 phase 2 duration: {result.s1_p2_durations}\n"
        f"Stage 1 phase 2 final costs: {result.s1_p2_final_costs}\n"

        f"Stage 2 status: {result.s2_status.name}\n"
        f"Stage 2 user time: {result.s2_user_time} seconds\n"
        f"Stage 2 system time: {result.s2_system_time} seconds\n"
        f"Stage 2 solve time: {result.s2_solve_time} seconds\n"
        f"Stage 2 time to best: {result.s2_time_to_best} seconds\n"
        f"Stage 2 sterations count: {result.s2_iterations_count}\n"
        f"Stage 2 stale iterations count: {result.s2_stale_iterations_count}\n"
        f"Stage 2 solution: {result.s2_solution or None}\n"
        f"Stage 2 solution cost: {result.s2_solution_cost}\n"
        f"Stage 2 phase 1 selected subsets: {result.s2_p1_selected_idxs}\n"
        f"Stage 2 phase 1 selected subsets count: {result.s2_p1_selected_idxs_count}\n"
        f"Stage 2 phase 1 shared cache hit count: {result.s2_p1_shared_cache_hit_count}\n"
        f"Stage 2 phase 1 shared cache miss count: {result.s2_p1_shared_cache_miss_count}\n"
        f"Stage 2 phase 1 repetitions: {result.s2_p1_repetitions}\n"
        f"Stage 2 phase 1 durations: {result.s2_p1_durations}\n"
        f"Stage 2 phase 1 final costs: {result.s2_p1_final_costs}\n"
        f"Stage 2 phase 2 (h,0) moves count: {result.s2_p2_h0_move_count}\n"
        f"Stage 2 phase 2 (h,1) moves count: {result.s2_p2_h1_move_count}\n"
        f"Stage 2 phase 2 (1,h) moves count: {result.s2_p2_1h_move_count}\n"
        f"Stage 2 phase 2 repetitions: {result.s2_p2_repetitions}\n"
        f"Stage 2 phase 2 improvements: {result.s2_p2_improvements}\n"
        f"Stage 2 phase 2 duration: {result.s2_p2_durations}\n"
        f"Stage 2 phase 2 final costs: {result.s2_p2_final_costs}\n"
    )

def build_worker_result_short_info(result: GraspWorkerResult) -> str:
    return (
        f">>> '{result.instance_id}' WORKER RESULT (pid: {result.pid}) <<<\n"
        f"Stages: {result.stages}\n"
        f"Status: {result.status.name}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Init time: {result.init_time} seconds\n"
        f"Inter stage time: {result.inter_stage_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Iterations count: {result.iterations_count}\n"
        f"Shared cache hit count: {result.p1_shared_cache_hit_count}\n"
        f"Shared cache miss count: {result.p1_shared_cache_miss_count}\n"
        f"Phase 2 (h,0) moves count: {result.p2_h0_move_count}\n"
        f"Phase 2 (h,1) moves count: {result.p2_h1_move_count}\n"
        f"Phase 2 (1,h) moves count: {result.p2_1h_move_count}\n"
    )

def build_stage_result_info(result: GraspStageResult) -> str:
    return (
        f">>> '{result.instance_id}' STAGE RESULT (pid: {result.pid}) <<<\n"
        f"Stage: {result.stage}\n"
        f"Status: {result.status.name}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Iterations count: {result.iterations_count}\n"
        f"Stale iterations count: {result.stale_iterations_count}\n"
        f"Shared cache hit count: {result.p1_shared_cache_hit_count}\n"
        f"Shared cache miss count: {result.p1_shared_cache_miss_count}\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"

        f"Phase 1 selected subsets: {result.p1_selected_idxs}\n"
        f"Phase 1 selected subsets count: {result.p1_selected_idxs_count}\n"
        f"Phase 1 shared cache hit count: {result.p1_shared_cache_hit_count}\n"
        f"Phase 1 shared cache miss count: {result.p1_shared_cache_miss_count}\n"
        f"Phase 1 repetitions: {result.p1_repetitions}\n"
        f"Phase 1 durations: {result.p1_durations}\n"
        f"Phase 1 final costs: {result.p1_final_costs}\n"
        f"Phase 2 (h,0) move count: {result.p2_h0_move_count}\n"
        f"Phase 2 (h,1) move count: {result.p2_h1_move_count}\n"
        f"Phase 2 (1,h) move count: {result.p2_1h_move_count}\n"
        f"Phase 2 repetitions: {result.p2_repetitions}\n"
        f"Phase 2 improvements: {result.p2_improvements}\n"
        f"Phase 2 duration: {result.p2_durations}\n"
        f"Phase 2 final costs: {result.p2_final_costs}\n"
    )

def build_stage_result_short_info(result: GraspStageResult) -> str:
    return (
        f">>> '{result.instance_id}' STAGE RESULT (pid: {result.pid}) <<<\n"
        f"Stage: {result.stage}\n"
        f"Status: {result.status.name}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Iterations count: {result.iterations_count}\n"
        f"Stale iterations count: {result.stale_iterations_count}\n"
        f"Shared cache hit count: {result.p1_shared_cache_hit_count}\n"
        f"Shared cache miss count: {result.p1_shared_cache_miss_count}\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
    )

def write_log(result: GraspResult) -> None:
    os.makedirs(result.log_file_path.parent, exist_ok=True)

    # Write json file
    json_file_path = result.log_file_path.with_suffix('.json')
    with open(json_file_path, 'w') as json_fp:
        json_fp.write(result.to_json())

    # Merge history of each worker
    events = []
    for subres in result.worker_results:
        events.extend(subres.s1_history)
        events.extend(subres.s2_history)
    events.sort()

    start = events[0][0] if events else 0
    history = [f"[{tmst - start:.2f}] {msg}" for tmst, msg in events]

    # Write log file
    log_lines = []
    log_lines.append(build_result_info(result))
    log_lines.append("\n>>>>> WORKER RESULTS <<<<<\n")
    log_lines.extend('\n'.join(map(build_worker_result_info, result.worker_results)))
    log_lines.append("\n>>>>> HISTORY <<<<<\n")
    log_lines.extend('\n'.join(history))

    with open(result.log_file_path, 'w') as fp:
        fp.writelines(log_lines)

    # Write profiler info (if profiling has been done)
    stats_file_path = result.log_file_path.with_suffix('.stats')
    worker_stats_dir_path = Path(stats_file_path.parent, stats_file_path.stem)
    if worker_stats_dir_path.exists():
        with open(stats_file_path, 'w') as fp:
            base_path, _, file_names = next(os.walk(worker_stats_dir_path))
            stats_file_paths = (os.path.join(base_path, fname) for fname in file_names)
            stats = Stats(*stats_file_paths, stream=fp)
            #stats.strip_dirs()
            stats.sort_stats('tottime')
            stats.print_stats()
            stats.print_callers()
