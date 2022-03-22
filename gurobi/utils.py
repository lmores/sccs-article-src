import os
import re

from utils.config import PROJECT_ROOT
from utils.datamodels import GurobiResult
from utils.misc import to_human_datetime
from utils.vprint import v0_print


ONE_INDEX_SUBSET_VAR_REGEXP = re.compile("^X\[(\d+)\]$")
TWO_INDEX_CONFLICT_VAR_REGEXP = re.compile("^Y\[(\d+),(\d+)\]$")

RC_HEADER_TEMPLATE = f"{'Var name':<20}{'Objective':>15}{'Reduced cost':>15}\n"
RC_ROW_TEMPLATE = "{var.VarName:<20}{var.X:>15.10f}{var.RC:>15.10f}\n"


def build_result_info(result: GurobiResult) -> str:
    return (
        f"\n>>>>> '{result.instance_id}' SOLUTION INFO <<<<<\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Solution valid: {result.solution_valid}\n"
        f"Solution best bound: {result.solution_best_bound}\n"
        f"Solution gap: {result.solution_gap}\n"
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

        f"\n>>>>> '{result.instance_id}' PARAMETERS INFO <<<<<\n"
        f"Time limit: {result.time_limit}{' seconds' if result.time_limit is not None else ''}\n"
        f"Relaxed: {result.relaxed}\n"

        f"\n>>>>> '{result.instance_id}' RUNTIME INFO <<<<<\n"
        f"Start at: {to_human_datetime(result.start_at)}\n"
        f"End at: {to_human_datetime(result.end_at)}\n"
        f"User time: {result.user_time}\n"
        f"System time: {result.system_time}\n"
        f"Read time: {result.read_time}\n"
        f"Init time: {result.init_time}\n"
        f"Time to best: {result.time_to_best}\n"
        f"Solve time: {result.solve_time}\n"
        f"Total time: {result.total_time}\n"

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
        f"Gurobi version: {result.gurobi_version}\n"
        f"Git hash: {result.git_hash}\n"
        f"Runtime exception: {result.runtime_exception}\n"
        # Note: 'result.results_file_path' and 'result.analyses_file_path' are
        # not available at this point as they are set in main.py
    )


def build_result_short_info(result: GurobiResult) -> None:
    return (
        f">>>>> '{result.instance_id}' (k={result.conflict_threshold}) GUROBI RESULT <<<<<\n"
        f"Status: {result.status.name}\n"
        f"Start at: {to_human_datetime(result.start_at)}\n"
        f"End at: {to_human_datetime(result.end_at)}\n"
        f"User time: {result.user_time} seconds\n"
        f"System time: {result.system_time} seconds\n"
        f"Init time: {result.init_time} seconds\n"
        f"Solve time: {result.solve_time} seconds\n"
        f"Time to best: {result.time_to_best} seconds\n"
        f"Solution: {result.solution}\n"
        f"Solution cost: {result.solution_cost}\n"
        f"Solution valid: {result.solution_valid}\n"
        f"Solution best bound: {result.solution_best_bound}\n"
        f"Solution gap: {result.solution_gap}\n"
        f"Solution subsets cost: {result.solution_subsets_cost}\n"
        f"Solution non zero subset costs count: {result.solution_non_zero_subset_costs_count}\n"
        f"Solution subset costs distrib: {result.solution_subset_costs_distrib}\n"
        f"Solution aggregated subset costs distrib: {result.solution_aggregated_subset_costs_distrib}\n"
        f"Solution conflicts cost: {result.solution_conflicts_cost}\n"
        f"Solution non zero conflict costs count: {result.solution_non_zero_conflict_costs_count}\n"
        f"Solution conflict costs distrib: {result.solution_conflict_costs_distrib}\n"
        f"Solution aggregated conflict costs distrib: {result.solution_aggregated_conflict_costs_distrib}\n"
    )


def write_log(result: GurobiResult, model) -> None:
    log_file_path = result.log_file_path
    gurobi_log_file_path = log_file_path.with_suffix('.gurobi.log')

    log_lines = [build_result_info(result), "\n>>>>> GUROBI OUTPUT <<<<<\n"]

    # Read temporary log file created by gurobi
    if gurobi_log_file_path.exists():
        with open(gurobi_log_file_path, 'r') as fp:
            log_lines.extend(fp.readlines())

    # Save log file
    os.makedirs(log_file_path.parent, exist_ok=True)
    with open(log_file_path, 'w') as fp:
        fp.writelines(log_lines)

        if result.relaxed:
            fp.write(f"\n\n{RC_HEADER_TEMPLATE}")
            fp.writelines(RC_ROW_TEMPLATE.format(var=var) for var in model.getVars())

    # Remove temporary log file created by gurobi (this operation may fail
    # if at this point the gurobi process is still locking the file for some
    # reason - this happens only on windows)
    try:
        gurobi_log_file_path.unlink()
    except OSError as e:
        v0_print("[WARNING] Failed to delete file '{}', remove it manually ({})",
            e.filename, e)
