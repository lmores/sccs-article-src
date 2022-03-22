from __future__ import annotations

import json
import os
import re
import sys
import subprocess

from itertools import combinations

from enum import IntEnum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, TypeVar

from utils.cmd_args import Action
from utils.misc import aggregate_distrib, build_distrib
from utils.vprint import v0_print

try:
    from gurobipy import gurobi
except ImportError:
    gurobi = None


PAIR_REGEX = re.compile(
    r'^\(\s*'
        r'(?P<x1>(?:\+|-)?(?:\d*.?\d+|\d+.?\d*))'
        r',\s*'
        r'(?P<x2>(?:\+|-)?(?:\d*.?\d+|\d+.?\d*))'
    r'\s*\)$'
)


# Helpers
def _get_python_version() -> str:
    name = sys.implementation.name
    v = sys.implementation.version
    return f"{name} {v.major}.{v.minor}.{v.micro}-{v.releaselevel}-{v.serial}"


def _get_gurobi_version() -> str:
    return '.'.join(map(str, gurobi.version())) if gurobi else ''


def _get_git_hash() -> str:
    try:
        cmd = ('git', 'log', '-n', '1', '--date=short', '--pretty=tformat:%h-%ad')
        return subprocess.check_output(cmd).strip().decode()
    except Exception as ex:
        v0_print("[WARNING] Unable to detect git hash. {}", ex)
        return ''


class ResultStatus(IntEnum):
    SKIPPED = -2
    RUNTIME_EXCEPTION = -1
    DONE = 0

    # Gurobi status codes
    LOADED = 1
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    CUTOFF = 6
    ITERATION_LIMIT = 7
    NODE_LIMIT = 8
    TIME_LIMIT = 9
    SOLUTION_LIMIT = 10
    INTERRUPTED = 11
    NUMERIC = 12
    SUBOPTIMAL = 13
    INPROGRESS = 14
    USER_OBJ_LIMIT = 15


@dataclass
class InstanceData:
    id: str
    n_elements: int
    n_subsets: int
    subsets: list[set[int]]
    subset_costs: list[int]
    conflict_costs: dict[(int,int),int]
    conflict_threshold: int


@dataclass
class BaseResult:
    # Main info
    instance_id: str
    action: Action
    status: ResultStatus = ResultStatus.LOADED
    cmd_args: list[str] = field(default_factory=list)
    cpu_count: int = os.cpu_count()
    time_limit: Optional[int] = None

    # Instance info
    n_elements: int = None
    n_subsets: int = None
    conflict_threshold: int = None

    subsets_cost: int = None
    non_zero_subset_costs_count: int = None
    subset_costs_distrib: dict[float,int] = None
    aggregated_subset_costs_distrib: dict[(int,int),int] = None

    conflicts_cost: int = None
    non_zero_conflict_costs_count: int = None
    conflict_costs_distrib: dict[float,int] = None
    aggregated_conflict_costs_distrib: dict[(int,int),int] = None
    max_conflicts: int = None

    # Runtime info
    start_at: float = 0
    end_at: float = 0
    user_time: float = 0
    system_time: float = 0
    read_time: float = 0
    total_time: float = 0

    # Other info
    log_file_path: Path = None
    results_file_path: Path = None
    analyses_file_path: Path = None
    python_interpreter: str = _get_python_version()
    git_hash: str = _get_git_hash()
    runtime_exception: Optional[Exception] = None

    @staticmethod
    def for_action(instance_id: str, action: Action, **kwargs) -> ExtendSccsBaseResult:
        if action == Action.SCCS_BIN or action == Action.SCCS_MIP:
            return GurobiResult(instance_id, action, **kwargs)
        elif action == Action.SCCS_GRASP:
            return GraspResult(instance_id, action, **kwargs)
        elif action == Action.ANALYSE:
            return AnalysisResult(instance_id, action, **kwargs)
        else:
            raise ValueError(f"Unhandled action for result object: {action}")

    def add_instance_data(self, data: InstanceData) -> None:
        self.n_elements = data.n_elements
        self.n_subsets = data.n_subsets
        self.conflict_threshold = data.conflict_threshold
        self.subsets_cost = sum(data.subset_costs)
        self.conflicts_cost = sum(c for ((i,j),c) in data.conflict_costs.items() if i < j)
        self.max_conflicts = self.n_subsets * (self.n_subsets - 1) // 2

        # Subset costs distrib
        self.subset_costs_distrib = build_distrib(data.subset_costs)
        self.aggregated_subset_costs_distrib = aggregate_distrib(self.subset_costs_distrib)
        self.non_zero_subset_costs_count = (
            sum(self.subset_costs_distrib.values())
            - self.subset_costs_distrib.get(0, 0)
        )

        # Conflict costs distrib
        conflict_costs = tuple(data.conflict_costs.get((i,j), 0)
                               for i,j in combinations(range(self.n_subsets), 2))
        self.conflict_costs_distrib = build_distrib(conflict_costs)
        self.aggregated_conflict_costs_distrib = aggregate_distrib(self.conflict_costs_distrib)
        self.non_zero_conflict_costs_count = (
            sum(self.conflict_costs_distrib.values())
            - self.conflict_costs_distrib.get(0, 0)
        )

@dataclass
class SolutionAnalysisResult():
    valid: bool = None

    subsets_cost: int = None
    non_zero_subset_costs_count: int = None
    subset_costs_distrib: int = None
    aggregated_subset_costs_distrib: int = None

    conflicts_cost: int = None
    non_zero_conflict_costs_count: int = None
    conflict_costs_distrib: dict[float,int] = None
    aggregated_conflict_costs_distrib: dict[(int,int),int] = None


ExtendSccsBaseResult = TypeVar('ExtendSccsBaseResult', bound='SccsBaseResult')

@dataclass
class SccsBaseResult(BaseResult):
    # Solution info
    solution: list[int] = None
    solution_cost: int = None
    solution_valid: bool = None

    solution_subsets_cost: int = None
    solution_non_zero_subset_costs_count: int = None
    solution_subset_costs_distrib: int = None
    solution_aggregated_subset_costs_distrib: int = None

    solution_conflicts_cost: int = None
    solution_non_zero_conflict_costs_count: int = None
    solution_conflict_costs_distrib: dict[float,int] = None
    solution_aggregated_conflict_costs_distrib: dict[(int,int),int] = None

    # Runtime info
    init_time: float = 0
    solve_time: float = 0
    time_to_best: float = 0

    def add_solution_info(self, info: SolutionAnalysisResult) -> None:
        self.solution_valid = info.valid

        self.solution_subsets_cost = info.subsets_cost
        self.solution_non_zero_subset_costs_count = info.non_zero_subset_costs_count
        self.solution_subset_costs_distrib = info.subset_costs_distrib
        self.solution_aggregated_subset_costs_distrib = info.aggregated_subset_costs_distrib

        self.solution_conflicts_cost = info.conflicts_cost
        self.solution_non_zero_conflict_costs_count = info.non_zero_conflict_costs_count
        self.solution_conflict_costs_distrib = info.conflict_costs_distrib
        self.solution_aggregated_conflict_costs_distrib = info.aggregated_conflict_costs_distrib

    _STR_FIELDS = frozenset((
        'instance_id', 'cmd_args', 'start_at', 'end_at',
        'python_interpreter', 'git_hash', 'runtime_exception'
    ))

    _INT_FIELDS = frozenset((
        'cpu_count', 'n_elements', 'n_subsets', 'conflict_threshold',
        'non_zero_subset_costs_count', 'non_zero_conflict_costs_count',
        'max_conflicts', 'solution_non_zero_subset_costs_count',
        'solution_non_zero_conflict_costs_count',
    ))

    _FLOAT_FIELDS = frozenset((
        'time_limit', 'subsets_cost', 'conflicts_cost',
        'user_time', 'system_time', 'read_time', 'total_time',
        'solution_cost', 'solution_subsets_cost', 'solution_conflicts_cost',
        'init_time', 'solve_time', 'time_to_best'
    ))

    _PLAIN_DICT_FIELDS = frozenset((
        'subset_costs_distrib', 'conflict_costs_distrib',
        'solution_subset_costs_distrib', 'solution_conflict_costs_distrib'
    ))

    _TUPLE_KEY_DICT_FIELDS = frozenset((
        'aggregated_subset_costs_distrib',
        'aggregated_conflict_costs_distrib',
        'solution_aggregated_subset_costs_distrib',
        'solution_aggregated_conflict_costs_distrib'
    ))

    _BOOLEAN_FIELDS = frozenset(('solution_valid',))
    _ENUM_STATUS_FIELDS = frozenset(('status',))
    _ENUM_ACTION_FIELD = frozenset(('action',))
    _PLAIN_LIST_FIELDS = frozenset(('solution',))
    _PATH_FIELDS = frozenset(('log_file_path', 'results_file_path', 'analyses_file_path'))

    @classmethod
    def from_dict(cls: ExtendSccsBaseResult, d: dict[str, str]) -> ExtendSccsBaseResult:
        kwargs = {}
        for fname, value in d.items():
            fname = fname.lower()
            if value == None or value == '':
                kwargs[fname] = value
            else:
                for field_category, field_ctor in cls._FIELD_CTORS:
                    if fname in field_category:
                        try:
                            kwargs[fname] = field_ctor(value)
                            break
                        except Exception as e:
                            print(
                                f"[ERROR] {field_ctor} could not build field "
                                f"'{fname}': {value}, type: {type(value)} "
                                f"({e.__cause__})"
                            )
                            kwargs[fname] = None
                else:
                    print(f"Unhandled field type for field name '{fname}' "
                          f"(value={value}, type={type(value)}")
                    kwargs[fname] = value

        instance_id = kwargs.pop('instance_id')
        action = kwargs.pop('action')
        return cls(instance_id, action, **kwargs)

    @classmethod
    def from_json(cls: ExtendSccsBaseResult, payload: str) -> ExtendSccsBaseResult:
        kwargs = json.loads(payload)
        for fname in cls._TUPLE_KEY_DICT_FIELDS:
            tmp = {}
            for k,v in kwargs[fname].items():
                m = PAIR_REGEX.match(k)
                t = (float(m.group('x1')), float(m.group('x2')))
                tmp[t] = v
            kwargs[fname] = tmp

        for fname in cls._PATH_FIELDS:
            kwargs[fname] = Path(kwargs[fname])

        return cls(**kwargs)


@dataclass
class GurobiResult(SccsBaseResult):
    relaxed: bool = False
    solution_best_bound: float = None
    solution_gap: float = None
    gurobi_version: str = _get_gurobi_version()

    _BOOLEAN_FIELDS = SccsBaseResult._BOOLEAN_FIELDS.union(('relaxed',))
    _FLOAT_FIELDS = SccsBaseResult._FLOAT_FIELDS.union(('solution_best_bound', 'solution_gap'))
    _STR_FIELDS = SccsBaseResult._STR_FIELDS.union(('gurobi_version',))
    _FIELD_CTORS = (
        (_STR_FIELDS, str),
        (SccsBaseResult._INT_FIELDS, int),
        (_FLOAT_FIELDS, float),
        (SccsBaseResult._PLAIN_LIST_FIELDS, eval),
        (SccsBaseResult._PLAIN_DICT_FIELDS, eval),
        (SccsBaseResult._TUPLE_KEY_DICT_FIELDS, eval),
        (_BOOLEAN_FIELDS, lambda x: x.lower() == 'true'),
        (SccsBaseResult._ENUM_ACTION_FIELD, lambda x: Action[x]),
        (SccsBaseResult._ENUM_STATUS_FIELDS, lambda x: ResultStatus[x]),
        (SccsBaseResult._PATH_FIELDS, Path),
    )

@dataclass
class GraspResult(SccsBaseResult):

    # Main info
    worker_count: int = None

    # Parameters info
    stages: int = None
    max_candidates: int = None
    presolve: float = None
    iterations_limit: int = None
    stale_iterations_limit: int = None

    # Parameters info for stage 1
    s1_max_candidates: int | None = None
    s1_time_limit: int | None = None
    s1_iterations_limit: int | None = None
    s1_stale_iterations_limit: int | None = None

    # Parameters info for stage 2
    s2_max_candidates: int | None = None
    s2_time_limit: int | None = None
    s2_iterations_limit: int | None = None
    s2_stale_iterations_limit: int | None = None

    # Runtime info
    user_time: float = 0
    system_time: float = 0
    presolve_time: float = 0
    inter_stage_time: float = 0

    # Execution info
    iterations_count: int = 0
    p1_shared_cache_hit_count: int = 0
    p1_shared_cache_miss_count: int = 0
    p2_h0_move_count: int = 0
    p2_h1_move_count: int = 0
    p2_1h_move_count: int = 0

    # Stage 1
    s1_status: ResultStatus = ResultStatus.LOADED
    s1_solution: set[int] = field(default_factory=set)
    s1_solution_cost: int = None
    s1_time_to_best: float = None

    # Stage 2
    s2_status: ResultStatus = ResultStatus.LOADED
    s2_solution: set[int] = field(default_factory=set)
    s2_solution_cost: int = None
    s2_time_to_best: float = None

    # Worker info
    worker_results: list[GraspWorkerResult] = field(default_factory=list)

    def add_worker_results(self, worker_results: list[GraspWorkerResult], data: InstanceData):
        from utils.analysis import analyse_solution

        self.worker_results = worker_results
        self.status = ResultStatus.DONE
        self.s1_status = ResultStatus.DONE
        self.s2_status = ResultStatus.DONE if self.stages == 2 else ResultStatus.SKIPPED
        for result in worker_results:
            self.user_time += result.user_time
            self.system_time += result.system_time
            self.iterations_count += result.iterations_count
            self.p1_shared_cache_hit_count = result.p1_shared_cache_hit_count
            self.p1_shared_cache_miss_count = result.p1_shared_cache_miss_count
            self.p2_h0_move_count += result.p2_h0_move_count
            self.p2_h1_move_count += result.p2_h1_move_count
            self.p2_1h_move_count += result.p2_1h_move_count

            # Result status
            if result.status == ResultStatus.INFEASIBLE:
                self.status == ResultStatus.INFEASIBLE
            elif result.status == ResultStatus.TIME_LIMIT:
                if self.status == ResultStatus.DONE:
                    self.status == ResultStatus.TIME_LIMIT

            if ((self.solution_cost is None or
                    result.solution_cost < self.solution_cost) or
                (result.solution_cost == self.solution_cost and
                    result.time_to_best < self.time_to_best)):

                self.solution = result.solution
                self.solution_cost = result.solution_cost
                self.time_to_best = result.time_to_best
                self.init_time = result.init_time

            # Stage 1 status
            if result.s1_status == ResultStatus.INFEASIBLE:
                self.s1_status == ResultStatus.INFEASIBLE
            elif result.s1_status == ResultStatus.TIME_LIMIT:
                if self.s1_status == ResultStatus.DONE:
                    self.s1_status == ResultStatus.TIME_LIMIT

            if ((self.s1_solution_cost is None or
                    result.s1_solution_cost < self.s1_solution_cost) or
                (result.s1_solution_cost == self.s1_solution_cost and
                    result.s1_time_to_best < self.s1_time_to_best)):

                self.s1_solution = result.s1_solution
                self.s1_solution_cost = result.s1_solution_cost
                self.s1_time_to_best = result.s1_time_to_best

            # Stage 2 status
            if result.s2_status == ResultStatus.INFEASIBLE:
                self.s2_status == ResultStatus.INFEASIBLE
            elif result.s2_status == ResultStatus.TIME_LIMIT:
                if self.s2_status == ResultStatus.DONE:
                    self.s2_status == ResultStatus.TIME_LIMIT

            if ((self.s2_solution_cost is None or
                    result.s2_solution_cost < self.s2_solution_cost) or
                (result.s2_solution_cost == self.s2_solution_cost and
                    result.s2_time_to_best < self.s2_time_to_best)):

                self.s2_solution = result.s2_solution
                self.s2_solution_cost = result.s2_solution_cost
                self.s2_time_to_best = result.s2_time_to_best

        self.solution = sorted(self.solution)
        solution_info = analyse_solution(self.solution, self.solution_cost, data)
        self.add_solution_info(solution_info)

    def to_json(self, **kwargs):
        d = self.__dict__.copy()
        for fname in self.__class__._TUPLE_KEY_DICT_FIELDS:
            d[fname] = {repr(k):v for k,v in getattr(self, fname).items()}

        def _default(obj):
            if isinstance(obj, set):
                return str(obj)
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, ResultStatus):
                return obj.name
            elif isinstance(obj, GraspWorkerResult):
                return obj.__dict__
            else:
                return obj

        return json.dumps(d, default=_default, **kwargs)

    _INT_FIELDS = SccsBaseResult._INT_FIELDS.union((
        'worker_count', 'stages', 'max_candidates', 'presolve', 'iterations_limit',
        'stale_iterations_limit',
        's1_max_candidates', 's1_iterations_limit', 's1_stale_iterations_limit',
        's2_max_candidates', 's2_iterations_limit', 's2_stale_iterations_limit',
        'iterations_count',
        'p1_shared_cache_hit_count', 'p1_shared_cache_miss_count',
        'p2_h0_move_count', 'p2_h1_move_count', 'p2_1h_move_count',
        's1_solution_cost', 's2_solution_cost'
    ))

    _FLOAT_FIELDS = SccsBaseResult._FLOAT_FIELDS.union((
        'presolve_time', 'inter_stage_time',
        's1_time_limit', 's1_time_to_best',
        's2_time_limit', 's2_time_to_best'
    ))
    _ENUM_STATUS_FIELDS = SccsBaseResult._ENUM_STATUS_FIELDS.union(('s1_status', 's2_status'))
    _PLAIN_LIST_FIELDS = SccsBaseResult._PLAIN_LIST_FIELDS.union(('s1_solution', 's2_solution'))
    _FIELD_CTORS = (
        (SccsBaseResult._STR_FIELDS, str),
        (_INT_FIELDS, int),
        (_FLOAT_FIELDS, float),
        (_PLAIN_LIST_FIELDS, eval),
        (SccsBaseResult._PLAIN_DICT_FIELDS, eval),
        (SccsBaseResult._TUPLE_KEY_DICT_FIELDS, eval),
        (SccsBaseResult._BOOLEAN_FIELDS, lambda x: x.lower() == 'true'),
        (SccsBaseResult._ENUM_ACTION_FIELD, lambda x: Action[x]),
        (_ENUM_STATUS_FIELDS, lambda x: ResultStatus[x]),
        (SccsBaseResult._PATH_FIELDS, Path),
    )


@dataclass
class GraspWorkerResult:
    # Global info
    instance_id: str
    pid: int
    stages: int
    status: ResultStatus = ResultStatus.LOADED
    log_file_path: Path | None = None

    # Parameters info
    s1_max_candidates: int | None = None
    s1_time_limit: int | None = None
    s1_iterations_limit: int | None = None
    s1_stale_iterations_limit: int | None = None

    s2_max_candidates: int | None = None
    s2_time_limit: int | None = None
    s2_iterations_limit: int | None = None
    s2_stale_iterations_limit: int | None = None

    # Runtime info
    user_time: float = 0
    system_time: float = 0
    init_time: float = 0
    inter_stage_time: float = 0
    time_to_best: float = 0
    solve_time: float = 0

    # Solution info
    solution: set[int] = field(default_factory=set)
    solution_cost: int = 0

    # Execution info
    iterations_count: int = 0
    p1_shared_cache_hit_count: int = 0
    p1_shared_cache_miss_count: int = 0
    p2_h0_move_count: int = 0
    p2_h1_move_count: int = 0
    p2_1h_move_count: int = 0

    # First stage runtime info
    s1_user_time: float = 0
    s1_system_time: float = 0
    s1_solve_time: float = 0
    s1_time_to_best: float = 0

    # First stage solution info
    s1_solution: set[int] = field(default_factory=set)
    s1_solution_cost: int = 0

    # First stage execution info
    s1_status: ResultStatus = ResultStatus.LOADED
    s1_history: list[str] = field(default_factory=list)
    s1_iterations_count: int = 0
    s1_stale_iterations_count: int = 0

    s1_p1_selected_idxs: list[int] = field(default_factory=list)
    s1_p1_selected_idxs_count: int = 0
    s1_p1_shared_cache_hit_count: int = 0
    s1_p1_shared_cache_miss_count: int = 0
    s1_p1_repetitions: list[int] = field(default_factory=list)
    s1_p1_durations: list[float] = field(default_factory=list)
    s1_p1_final_costs: list[int] = field(default_factory=list)

    s1_p2_h0_move_count: list[int] = field(default_factory=list)
    s1_p2_h1_move_count: list[int] = field(default_factory=list)
    s1_p2_1h_move_count: list[int] = field(default_factory=list)
    s1_p2_repetitions: list[int] = field(default_factory=list)
    s1_p2_improvements: list[int] = field(default_factory=list)
    s1_p2_durations: list[float] = field(default_factory=list)
    s1_p2_final_costs: list[int] = field(default_factory=list)

    # Second stage runtime info
    s2_user_time: float = 0
    s2_system_time: float = 0
    s2_solve_time: float = 0
    s2_time_to_best: float = 0

    # Second stage solution info
    s2_solution: set[int] = field(default_factory=set)
    s2_solution_cost: int = 0

    # Second stage execution info
    s2_status: ResultStatus = ResultStatus.LOADED
    s2_history: list[str] = field(default_factory=list)
    s2_iterations_count: int = 0
    s2_stale_iterations_count: int = 0

    s2_p1_selected_idxs: list[int] = field(default_factory=list)
    s2_p1_selected_idxs_count: int = 0
    s2_p1_shared_cache_hit_count: int = 0
    s2_p1_shared_cache_miss_count: int = 0
    s2_p1_repetitions: list[int] = field(default_factory=list)
    s2_p1_durations: list[float] = field(default_factory=list)
    s2_p1_final_costs: list[int] = field(default_factory=list)

    s2_p2_h0_move_count: list[int] = field(default_factory=list)
    s2_p2_h1_move_count: list[int] = field(default_factory=list)
    s2_p2_1h_move_count: list[int] = field(default_factory=list)
    s2_p2_repetitions: list[int] = field(default_factory=list)
    s2_p2_improvements: list[int] = field(default_factory=list)
    s2_p2_durations: list[float] = field(default_factory=list)
    s2_p2_final_costs: list[int] = field(default_factory=list)

    def add_first_stage_info(self, info: GraspStageResult):
        if self.s2_status is not ResultStatus.LOADED:
            raise ValueError("Stage 1 info cannot be added after stage 2 info")
        elif self.pid != info.pid:
            raise ValueError("Cannot add stage info of a different worker. "
                            f"Worker pid: {self.pid}, stage pid: {info.pid}")

        _fnames = set(f.name for f in fields(self))
        for field in fields(info):
            _fname = f's1_{field.name}'
            if _fname in _fnames:
                setattr(self, _fname, getattr(info, field.name))

        self.status = info.status
        self.user_time = info.user_time
        self.system_time = info.system_time
        self.solve_time = info.solve_time
        self.time_to_best = info.time_to_best
        self.solution = info.solution
        self.solution_cost = info.solution_cost
        self.iterations_count = info.iterations_count
        self.p1_shared_cache_hit_count = info.p1_shared_cache_hit_count
        self.p1_shared_cache_miss_count = info.p1_shared_cache_miss_count
        self.p2_h0_move_count = info.p2_h0_move_count
        self.p2_h1_move_count = info.p2_h1_move_count
        self.p2_1h_move_count = info.p2_1h_move_count

    def add_second_stage_info(self, info: 'GraspStageResult', elapsed_time: int):
        if self.s1_status is ResultStatus.LOADED:
            raise ValueError("Cannot add stage 2 info without stage 1 info")
        elif self.pid != info.pid:
            raise ValueError("Cannot add stage info of a different worker. "
                            f"Worker pid: {self.pid}, stage pid: {info.pid}")

        _fnames = set(f.name for f in fields(self))
        for field in fields(info):
            _fname = f's2_{field.name}'
            if _fname in _fnames:
                setattr(self, _fname, getattr(info, field.name))

        stage_statuses = (self.s1_status, info.status)
        if ResultStatus.INFEASIBLE in stage_statuses:
            self.status = ResultStatus.INFEASIBLE
        elif ResultStatus.TIME_LIMIT in stage_statuses:
            self.status = ResultStatus.TIME_LIMIT
        elif all(status == ResultStatus.DONE for status in stage_statuses):
            self.status = ResultStatus.DONE
        else:
            raise RuntimeError(f"Unhandled stage status: {stage_statuses}")

        self.user_time += info.user_time
        self.system_time += info.system_time

        # For two stages executions 'self.solve_time' is set externally (since
        # time take by inter process communication must be taken into account)

        if info.solution_cost < self.solution_cost:
            self.solution = info.solution
            self.solution_cost = info.solution_cost
            self.time_to_best = info.time_to_best + elapsed_time

        self.iterations_count += info.iterations_count
        self.p1_shared_cache_hit_count += info.p1_shared_cache_hit_count
        self.p1_shared_cache_miss_count += info.p1_shared_cache_miss_count
        self.p2_h0_move_count += info.p2_h0_move_count
        self.p2_h1_move_count += info.p2_h1_move_count
        self.p2_1h_move_count += info.p2_1h_move_count


@dataclass
class GraspStageResult:
    instance_id: str
    pid: int
    stage: int = 0
    status: ResultStatus = ResultStatus.LOADED

    # Runtime info
    user_time: float = 0
    system_time: float = 0
    solve_time: float = 0
    time_to_best: float = 0

    # Solution info
    solution: set[int] = field(default_factory=set)
    solution_cost: int = 0

    # Execution info
    history: list[str] = field(default_factory=list)
    iterations_count: int = 0
    stale_iterations_count: int = 0

    p1_selected_idxs: list[int] = field(default_factory=list)
    p1_selected_idxs_count: int = 0
    p1_shared_cache_hit_count: int = 0
    p1_shared_cache_miss_count: int = 0
    p1_repetitions: list[int] = field(default_factory=list)
    p1_durations: list[float] = field(default_factory=list)
    p1_final_costs: list[int] = field(default_factory=list)

    p2_h0_move_count: list[int] = field(default_factory=list)
    p2_h1_move_count: list[int] = field(default_factory=list)
    p2_1h_move_count: list[int] = field(default_factory=list)
    p2_repetitions: list[int] = field(default_factory=list)
    p2_durations: list[float] = field(default_factory=list)
    p2_improvements: list[int] = field(default_factory=list)
    p2_final_costs: list[int] = field(default_factory=list)


@dataclass
class AnalysisResult(BaseResult):
    _PATH_FIELDS = ('log_file_path', 'results_file_path', 'analyses_file_path')
    _TUPLE_AS_KEY_DICT_FIELDS = (
        'aggregated_subset_costs_distrib',
        'aggregated_conflict_costs_distrib',
        'aggregated_intersection_sizes_distrib'
    )

    row_sums: int = None
    row_avg: float = None
    row_stddev: float = None
    column_sums: int = None
    column_avg: float = None
    column_stddev: float = None

    #unique_elements: list[int] = None
    #connected_components_count: int = None
    #connected_component_sizes: list[int] = None
    non_empty_intersections_count: int = None
    intersection_sizes_distrib: dict[float,int] = None
    aggregated_intersection_sizes_distrib: dict[(int,int),int] = None

    def to_json(self, **kwargs):
        d = self.__dict__.copy()
        for fname in self.__class__._TUPLE_AS_KEY_DICT_FIELDS:
            d[fname] = {repr(k):v for k,v in getattr(self, fname).items()}

        def _default(obj):
            if isinstance(obj, set):
                return str(obj)
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        return json.dumps(d, default=_default, **kwargs)

    @classmethod
    def from_json(cls, payload: str) -> 'GraspResult':
        kwargs = json.loads(payload)
        for fname in cls._TUPLE_AS_KEY_DICT_FIELDS:
            tmp = {}
            for k,v in kwargs[fname].items():
                m = PAIR_REGEX.match(k)
                t = (float(m.group('x1')), float(m.group('x2')))
                tmp[t] = v
            kwargs[fname] = tmp

        for fname in cls._PATH_FIELDS:
            kwargs[fname] = Path(kwargs[fname])

        return GraspResult(**kwargs)
