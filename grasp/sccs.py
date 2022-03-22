import ctypes
import os
import sys

from itertools import repeat
from heapq import nlargest
from math import ceil, sqrt
from multiprocessing import Manager, Process, Value
from multiprocessing.queues import Queue
from pathlib import Path
from time import time

from gurobi.sccs import build_bin_model
from gurobi.utils import ONE_INDEX_SUBSET_VAR_REGEXP
from utils.cmd_args import Action
from utils.config import LOG_FILE_NAME_TEMPLATE, LOGS_DIR_PATH
from utils.datamodels import GraspResult, GraspStageResult, GraspWorkerResult, InstanceData
from utils.misc import current_human_datetime, to_human_datetime
from utils.vprint import get_verbosity_level, v0_print, v1_print

from .sccs_worker import run_one_stage_sccs, run_two_stage_sccs
from .utils import (
    MockSharedDict,
    build_stage_result_short_info,
    build_worker_result_short_info,
    write_log,
)


def run_sccs(data: InstanceData,
        max_candidates: int | None=None,
        time_limit: float | None=None,
        stage1_time_limit: float | None =None,
        stage2_time_limit: float | None=None,
        iterations_limit: int | None=None,
        stale_iterations_limit: int | None=None,
        cpu_count: int=os.cpu_count(),
        presolve: float | None=None,
        profile=False
    ) -> GraspResult:

    verbosity, silent = get_verbosity_level()

    # Setup
    if cpu_count < 1:
        raise ValueError(f"'cpu_count' argument must be >= 1, got {cpu_count}")
    elif (os_cpu_count := os.cpu_count()) < cpu_count:
        v0_print(
            "[WARNING] {0} CPUs has been requested; however, using more CPUs "
            "than the actual number available on this machine ({1}) is not "
            "allowed. The GRASP algorithm will run using {1} CPUs and {2} "
            "workers (one process is reserved for inter process communication)",
            cpu_count, os_cpu_count, os_cpu_count - 1 if os_cpu_count > 1 else os_cpu_count
        )
        cpu_count = os_cpu_count
    worker_count = cpu_count - 1 if cpu_count > 1 else cpu_count

    if (time_limit is None and stage1_time_limit is None and stage2_time_limit is None
            and iterations_limit is None and stale_iterations_limit is None):
        raise ValueError(
            "'time_limit', 'stage1_time_limit', 'stage1_time_limit', "
            "'iterations_limit' and 'stale_iterations_limit' are all None. "
            "Provide a positive value for at least one of them."
        )

    s1tl = stage1_time_limit
    s2tl = stage2_time_limit
    if time_limit is not None:
        if s1tl is not None and s2tl is not None and time_limit != s1tl + s2tl:
            v0_print(
                "[WARNING] 'time_limit' value ({}) has been overridden by "
                "'stage1_time_limit' and 'stage2_time_limit' values "
                "({} + {} = {})", time_limit, s1tl, s2tl, s1tl + s2tl
            )
            time_limit = s1tl + s2tl
        if s1tl is None and s2tl is not None:
            s1tl = time_limit - s2tl
        elif s1tl is not None and s2tl is None:
            s2tl = time_limit - s1tl
        else:
            s1tl = time_limit / 2
            s2tl = time_limit / 2

    if max_candidates is None:
        max_candidates = ceil(sqrt(data.n_subsets))

    if presolve is not None and presolve < 1:
        raise ValueError(f"'presolve' argument must be >= 1, got {presolve}")

    v0_print(
        f"\n{'-' * 80}\n"  "[{}] Solving SCCS instance '{}' using GRASP\n"
        "(n_elements={}, n_subsets={}, conflict_threshold={}, max_candidates={}, "
        "time_limit={}, stage1_time_limit={}, stage2_time_limit={}, "
        "iterations_limit={}, stale_iterations_limit={}, n_cpu={}, presolve={})\n"
        f"{'-' * 80}\n", current_human_datetime(), data.id, data.n_elements,
        data.n_subsets, data.conflict_threshold, max_candidates, time_limit,
        s1tl, s2tl, iterations_limit, stale_iterations_limit, cpu_count, presolve
    )

    start_at = time()
    log_file_path = Path(LOGS_DIR_PATH, LOG_FILE_NAME_TEMPLATE.format(
        base=os.path.splitext(data.id)[0],
        type='grasp',
        datetime=to_human_datetime(start_at)
    ))
    os.makedirs(log_file_path.parent, exist_ok=True)

    result = GraspResult(
        # Main info
        instance_id=data.id,
        action=Action.SCCS_GRASP,
        cmd_args=sys.argv,
        cpu_count=cpu_count,
        worker_count=worker_count,   # status is set at the end

        # Parameters info
        stages = 1 if (worker_count == 1 or presolve is not None) else 2,
        max_candidates=max_candidates,
        presolve=presolve,
        time_limit=time_limit,
        iterations_limit=iterations_limit,
        stale_iterations_limit=stale_iterations_limit,

        s1_max_candidates=max_candidates,
        s1_time_limit=s1tl,
        s1_iterations_limit=iterations_limit,
        s1_stale_iterations_limit=stale_iterations_limit,

        s2_max_candidates=max_candidates,
        s2_time_limit=s2tl,
        s2_iterations_limit=iterations_limit,
        s2_stale_iterations_limit=stale_iterations_limit,

        # Runtime info
        start_at=start_at,
        # 'end_at', '(user,system,presolve,init,total)_time' and 'time_to_best'
        # are set at the end of this function. 'read_time' is set externally.

        # Other
        log_file_path=log_file_path
    )
    result.add_instance_data(data)

    # Presolve
    selected_idxs = None
    if presolve:
        presolve_time_start = time()
        model = build_bin_model(data, relaxed=True)
        model.optimize()
        result.presolve_time = time() - presolve_time_start

        non_zero_var_count = 0
        subset_variable_scores = [None for _ in range(data.n_subsets)]
        for var in model.getVars():
            var_idx_match = ONE_INDEX_SUBSET_VAR_REGEXP.match(var.varName)
            if not var_idx_match:
                continue
            var_idx = int(var_idx_match.group(1))
            subset_variable_scores[var_idx] = (var.X, -abs(var.RC))
            if var.X != 0:
                non_zero_var_count += 1

        selected_idxs = nlargest(round(presolve * non_zero_var_count),
            range(data.n_subsets), key=lambda i: subset_variable_scores[i])

    # Solve
    solve_time_start = time()

    if worker_count == 1:
        shared_cache = MockSharedDict()
        incumbent: Value = Value(ctypes.c_ulong, sys.maxsize)
        worker_results = [
            run_one_stage_sccs(data, max_candidates, shared_cache, incumbent,
                selected_idxs, time_limit, iterations_limit,
                stale_iterations_limit, verbosity, silent, log_file_path, profile
            )
        ]

    elif presolve:
        with Manager() as manager:
            shared_cache = manager.dict()
            incumbent: Value = Value(ctypes.c_ulong, sys.maxsize)
            with manager.Pool(processes=cpu_count) as pool:
                async_results = [pool.apply_async(
                    run_one_stage_sccs, (data, max_candidates, shared_cache,
                        incumbent, selected_idxs, time_limit, iterations_limit,
                        stale_iterations_limit, verbosity, silent,
                        log_file_path, profile)
                    ) for _ in repeat(None, cpu_count)
                ]
                worker_results: list[GraspWorkerResult] = [res.get() for res in async_results]

    else:
        with Manager() as manager:
            incumbent: Value = Value(ctypes.c_ulong, sys.maxsize)
            shared_cache: dict = manager.dict()
            outbound_queue: Queue = manager.Queue()
            inbound_queue: Queue = manager.Queue()

            workers = [
                Process(target=run_two_stage_sccs, args=(data, max_candidates,
                    shared_cache, incumbent, outbound_queue, inbound_queue,
                    s1tl, s2tl, iterations_limit, stale_iterations_limit,
                    verbosity, silent, log_file_path, profile
                )) for _ in repeat(None, worker_count)
            ]

            for worker in workers:
                worker.start()

            counter = 0
            selected_idxs = set()
            inter_stage_time_start: int | None = None
            while counter < worker_count:
                stage1_result: GraspStageResult = outbound_queue.get()
                selected_idxs.update(stage1_result.solution)
                counter += 1
                if inter_stage_time_start is None:
                    inter_stage_time_start = time()

                # Display first stage solution info
                v0_print(
                    "[INFO] Received first stage result {}/{} from worker {}",
                    counter, worker_count, stage1_result.pid, flush=True
                )
                if verbosity > 0 and not silent:
                    msg = build_stage_result_short_info(stage1_result)
                    v1_print(msg, flush=True)

            shared_cache.clear()

            result.inter_stage_time = time() - inter_stage_time_start
            for i in range(1, worker_count + 1):
                v0_print(
                    "[INFO] Sending selected subsets to worker {}/{} for "
                    "the second stage", i, worker_count
                )
                inbound_queue.put(selected_idxs)

            counter = 0
            worker_results = []
            while counter < worker_count:
                stage2_result: GraspStageResult = outbound_queue.get()
                worker_results.append(stage2_result)
                counter += 1

                # Display second stage solution info
                v0_print(
                    "[INFO] Received second stage result {}/{} from worker {}",
                    counter, worker_count, stage2_result.pid, flush=True
                )
                if verbosity > 0 and not silent:
                    msg = build_worker_result_short_info(stage2_result)
                    v1_print(msg, flush=True)

            for worker in workers:
                worker.join()

    result.end_at = solve_time_end = time()
    result.solve_time = solve_time_end - solve_time_start
    result.total_time = result.end_at - result.start_at

    # Extract info from worker results
    result.add_worker_results(worker_results, data)

    # Save result
    write_log(result)

    return result
