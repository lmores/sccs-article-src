#!/usr/bin/env python3

import os
import sys

from itertools import repeat
from pathlib import Path
from time import time
from traceback import format_exc

from psutil import Process

import grasp
import gurobi

from utils import analysis
from utils.cmd_args import Action, get_arg_parser, print_args
from utils.config import ANALYSES_DIR_PATH, DATASETS_DIR_PATH, RESULTS_DIR_PATH
from utils.datamodels import (
    BaseResult,
    AnalysisResult,
    InstanceData,
    GraspResult,
    GurobiResult,
    ResultStatus,
)
from utils.instance import read_instance
from utils.misc import current_human_datetime, nat_sort_key
from utils.results import save_result
from utils.vprint import set_verbosity_level, v0_print


def _print_result(result: BaseResult) -> None:
    if isinstance(result, GraspResult):
        v0_print(grasp.utils.build_result_short_info(result))
    elif isinstance(result, GurobiResult):
        v0_print(gurobi.utils.build_result_short_info(result))
    elif isinstance(result, AnalysisResult):
        v0_print(analysis.build_analysis_info(result))


def _run_action(action: Action, data: InstanceData, cmd_args) -> BaseResult:
    result = None

    # Analyse instance
    if action == Action.ANALYSE:
        result = analysis.analyse_instance(data)

    # SCCS
    elif action == Action.SCCS_BIN:
        result = gurobi.sccs.run_sccs_bin(data, time_limit=cmd_args.time_limit,
            relaxed=cmd_args.gurobi_relaxed, save_model=cmd_args.gurobi_save_model)

    elif action == Action.SCCS_MIP:
        result = gurobi.sccs.run_sccs_mip(data, time_limit=cmd_args.time_limit,
            relaxed=cmd_args.gurobi_relaxed, save_model=cmd_args.gurobi_save_model)

    elif action == Action.SCCS_GRASP:
        result = grasp.sccs.run_sccs(data,
            max_candidates=cmd_args.grasp_max_candidates,
            time_limit=cmd_args.time_limit,
            stage1_time_limit=cmd_args.grasp_stage1_time_limit,
            stage2_time_limit=cmd_args.grasp_stage2_time_limit,
            iterations_limit=cmd_args.grasp_iterations_limit,
            stale_iterations_limit=cmd_args.grasp_stale_iterations_limit,
            cpu_count=cmd_args.cpu_count, presolve=cmd_args.grasp_presolve,
            profile=cmd_args.grasp_profile
        )

    else:
        raise ValueError(f"Unhandled action: {action}")

    return result


def _get_unique_dataset_sequence(cmd_args) -> str:
        path = Path(DATASETS_DIR_PATH, cmd_args.target)
        if path.is_file():
            yield cmd_args.target
        elif path.is_dir():
            for dir_path, _, file_names in os.walk(path):
                file_names.sort(key=nat_sort_key)
                for fname in file_names:
                    yield str(Path(dir_path, fname).relative_to(DATASETS_DIR_PATH))
                if not cmd_args.recursive:
                    break
        else:
            raise FileNotFoundError(f"File or directory '{path}' does not exist")


def _get_task_sequence(cmd_args) -> tuple[str, Action]:
    for instance_id in _get_unique_dataset_sequence(cmd_args):
        for action in cmd_args.actions:
            if action == Action.SCCS_GRASP:
                for _ in repeat(None, cmd_args.grasp_repetitions):
                    yield instance_id, action
            else:
                yield instance_id, action


def main():
    # Setup
    cmd_args = get_arg_parser().parse_args()
    set_verbosity_level(level=cmd_args.verbose, silent=cmd_args.silent)
    print_args(cmd_args)

    human_now = current_human_datetime()
    results_file_path = Path(RESULTS_DIR_PATH, f"results_{human_now}.csv")
    analyses_file_path = Path(ANALYSES_DIR_PATH, f"analyses_{human_now}.csv")

    # Execute actions
    for instance_id, action in _get_task_sequence(cmd_args):
        result = BaseResult.for_action(instance_id, action, cmd_args=sys.argv)

        # Read instance data
        k: int = cmd_args.conflict_threshold
        v0_print("[INFO] Reading instance '{}' (k={}) from file", instance_id, k)
        read_time_start = time()
        try:
            instance_data = read_instance(instance_id, k)
        except Exception as ex:
            v0_print(
                "[ERROR] The following exception occurred while reading data "
                "for instance '{}'.\n{}", instance_id, format_exc()
            )
            result.status = ResultStatus.RUNTIME_EXCEPTION
            result.runtime_exception = ex
            save_result(result)
            continue
        read_time = time() - read_time_start

        # Execute action
        process = Process()
        cpu_times_start = process.cpu_times()
        start_at = time()
        try:
            result = _run_action(action, instance_data, cmd_args)
        except Exception as ex:
            cpu_times_end = process.cpu_times()
            v0_print(
                "[ERROR] The following exception occurred while running "
                "action '{}' for instance '{}' (k={}).\n{}",
                action, instance_id, k, format_exc()
            )
            result.add_instance_data(instance_data)
            result.start_at = start_at
            result.end_at = time()
            result.status = ResultStatus.RUNTIME_EXCEPTION
            result.runtime_exception = ex
            result.user_time = cpu_times_end.user - cpu_times_start.user
            result.system_time = cpu_times_end.system - cpu_times_start.system
            result.total_time = result.end_at - result.start_at
        finally:
            result.read_time = read_time
            result.results_file_path = results_file_path
            result.analyses_file_path = analyses_file_path

            saved = save_result(result)
            if saved:
                _print_result(result)


if __name__ == '__main__':
    main()
