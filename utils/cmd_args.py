from __future__ import annotations

import argparse
import enum
import os
from pathlib import Path

from typing import Union

from utils.config import DATASETS_DIR_PATH, LOGS_DIR_PATH, PROJECT_ROOT
from utils.vprint import v0_print


RELATIVE_LOGS_DIR_PATH = Path(LOGS_DIR_PATH).relative_to(PROJECT_ROOT)
RELATIVE_DATASET_DIR_PATH = Path(DATASETS_DIR_PATH).relative_to(PROJECT_ROOT)


# Helpers
def _int_or_none(x: str) -> Union[int, None]:
    return None if x == 'None' else int(x)


# Public classes and functions
class Action(str, enum.Enum):
    ANALYSE = 'analyse'
    SCCS_BIN = 'sccs_bin'
    SCCS_GRASP = 'sccs_grasp'
    SCCS_MIP = 'sccs_mip'

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def argparse(name) -> Action:
        try:
            return Action[name.upper()]
        except KeyError as e:
            raise argparse.ArgumentTypeError(e)


def print_args(args) -> None:
    lines = ["[INFO] Running with the following arguments:\n"]
    lines.extend(f"{key:32}: {val}\n" for key, val in vars(args).items())
    lines.append('\n')
    v0_print(''.join(lines))


def get_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)

    # Common args
    common_group = arg_parser.add_argument_group(title="Common arguments",
        description="arguments that affect all (or most) types of actions")

    common_group.add_argument('target', type=str, metavar="TARGET",
        help="relative path to a dataset file or directory thereof "
            f"inside the '{RELATIVE_DATASET_DIR_PATH}/' folder "
            "(e.g.: 'beasley/dataset.txt')")

    common_group.add_argument('-a', '--actions', nargs='*',
        type=Action.argparse, choices=tuple(Action),
        default=[Action.SCCS_GRASP], metavar="ACTION",
        help="the list of actions to execute for each dataset, chosen among "
            "(case insensitive): ANALYSE, SCCS_BIN, SCCS_MIP, "
            "SCCS_GRASP (default)")

    common_group.add_argument('-c', '--cpu-count',
        type=int, default=os.cpu_count(), metavar='CPU_COUNT',
        help="number of available cpu to use (default: the number of "
            f"logical cores available on this machine, i.e. {os.cpu_count()}). "
            "When CPU_COUNT > 1 the GRASP algorithm will always use at most "
            "CPU_COUNT-1 worker processes as one process is reserved to "
            "handle inter process communication")

    common_group.add_argument("-h", "--help", action="help",
        help="show this help message and exit")

    common_group.add_argument('-k', '--conflict-threshold',
        type=int, default=1, help="threshold above which two sets are "
            "considered to be in conflict (default: 1)")

    common_group.add_argument('-r', '--recursive', action="store_true",
        help="if TARGET is a directory, recursively solve all instances "
            "in that directory and its subdirectories (default: false)")

    common_group.add_argument('-s', '--silent', action="store_true",
        help="completely turns off output on the standard output "
            "and ignores verbosity level (default: false)")

    common_group.add_argument('-t', '--time-limit', type=float, default=None,
        help="amount of time (in seconds) after which the execution "
            "is halted (default: None). When this value is not none and the "
            "2-stage GRASP algorithm is executed, each stage has a time limit "
            "equal to half the value of this argument (unless "
            "'--grasp-stage1-time-limit' or '--grasp-stage2-time-limit') "
            "have been specified")

    common_group.add_argument('-v', '--verbose', action='count', default=0,
        help="set the verbosity of the output on standard output. "
            "Available levels: 0 (default), 1 (-v) or 2 (-vv). "
            "Higher levels prints more information.")

    # Grasp specific args
    grasp_group = arg_parser.add_argument_group(
        title="Grasp algorithm arguments",
        description="Arguments that affect only the SCCS_GRASP actions")

    grasp_group.add_argument('--grasp-iterations-limit',
        type=int, default=None, metavar='ITERATIONS_LIMIT',
        help="execution is halted after ITERATIONS_LIMIT iterations "
            "(default: None)")

    grasp_group.add_argument('--grasp-max-candidates', type=int, default=None,
        metavar="MAX_CANDIDATES", help="number of best candidates among which "
            "one is chosen at random in the first phase of the GRASP "
            "algorithm (default: ceil(sqrt(n_subsets)))")

    grasp_group.add_argument('--grasp-stage1-time-limit', type=float, default=None,
        metavar="TIME_LIMIT", help="amount of time (in seconds) after which "
            "the execution is halted during the first stage of the "
            "GRASP algorithm (default: None).")

    grasp_group.add_argument('--grasp-stage2-time-limit', type=float, default=None,
        metavar="TIME_LIMIT", help="amount of time (in seconds) after which "
            "the execution is halted during the second stage of the "
            "GRASP algorithm (default: None)")

    grasp_group.add_argument('--grasp-presolve', type=float, default=None,
            metavar="Z", help="subsets available for the construction of a "
            "feasible solution during the first phase of the GRASP algorithm "
            "are limited to Z * N, where N is the number of non-zero variables "
            "in the solution of the relaxed LP model and subsets are sorted "
            "descendingly according to (var, -|reduced_cost|)")

    grasp_group.add_argument('--grasp-profile', action='store_true',
        help="each worker process that executes the algorithm will be profiled "
            f"and results will be stored in the '{RELATIVE_LOGS_DIR_PATH}/' "
            "directory")

    grasp_group.add_argument('--grasp-repetitions', type=int, default=1,
        metavar="REPETITIONS", help="number of times the GRASP algorithm is "
            "repeated (default: 1). Since GRASP is not a deterministic "
            "algorithm it can be useful to gather information on the outcome "
            "of multiple executions.")

    grasp_group.add_argument('--grasp-stale-iterations-limit',
        type=_int_or_none, default=50, metavar='STALE_ITERATIONS_LIMIT',
        help="execution is halted if the incumbent solution has not "
            "improved since STALE_ITERATIONS_LIMIT iterations (default: 50)")

    # Gurobi specific args
    gurobi_group = arg_parser.add_argument_group(title="Gurobi arguments",
        description="Arguments that affect only SCCS_BIN and SCCS_MIP actions")

    gurobi_group.add_argument('--gurobi-relaxed', action="store_true",
        help="the associated LP problem is solved instead of the integer one "
            "when performing a SCCS_BIN or SCCS_MIP action (default: false).")

    gurobi_group.add_argument('--gurobi-save-model', action="store_true",
        help="save the model description in a file with extension '.lp' "
            "(default: false). Warning: model files may be huge (>1GB)")

    return arg_parser
