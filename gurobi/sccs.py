import os
import sys

from pathlib import Path
from psutil import Process
from time import time

import gurobipy as gp
from gurobipy import GRB

from utils.analysis import analyse_solution
from utils.cmd_args import Action
from utils.config import LOG_FILE_NAME_TEMPLATE, LOGS_DIR_PATH
from utils.datamodels import GurobiResult, InstanceData, ResultStatus
from utils.misc import current_human_datetime, to_human_datetime
from utils.vprint import v0_print

from .utils import ONE_INDEX_SUBSET_VAR_REGEXP, TWO_INDEX_CONFLICT_VAR_REGEXP, write_log


### BIN model
def build_bin_model(data: InstanceData, relaxed=False):
    model = gp.Model(f'{data.id}_sccs_bin')

    SC = gp.tupledict(enumerate(data.subset_costs))
    CC = gp.tupledict(((i,j),v) for ((i,j),v) in data.conflict_costs.items() if i < j)

    X = model.addVars(SC.keys(), vtype=GRB.BINARY, name="X")
    Y = model.addVars(CC.keys(), vtype=GRB.BINARY, name="Y")

    model.setObjective(X.prod(SC) + Y.prod(CC), GRB.MINIMIZE)

    LinExpr = gp.LinExpr
    model.addConstrs(
        (LinExpr((1, X[j]) for j,s in enumerate(data.subsets) if i in s) >= 1
        for i in range(data.n_elements)), name="cov"
    )

    model.addConstrs(
        (LinExpr(((1, X[j]), (1, X[t]), (-1, Y[j,t]))) <= 1 for j,t in CC.keys()),
        name="conf" #name=f"conf[{j},{t}]"
    )

    if relaxed:
        _model = model
        model = model.relax()
        _model.dispose()

    model.update()

    return model


def run_sccs_bin(data: InstanceData, time_limit=None, relaxed=False,
        save_model=False) -> GurobiResult:

    v0_print(
        f"\n{'-' * 80}\n"  "[{}] Solving '{}' SCCS INSTANCE using BIN MODEL\n"
        "(n_elements={}, n_subsets={}, conflict_threshold={}, "
        "time_limit={}, relaxed={}, save_model={})\n"  f"{'-' * 80}",
        current_human_datetime(), data.id, data.n_elements, data.n_subsets,
        data.conflict_threshold, time_limit, relaxed, save_model
    )

    start_at = time()
    log_file_path = Path(LOGS_DIR_PATH, LOG_FILE_NAME_TEMPLATE.format(
        base=os.path.splitext(data.id)[0],
        type='bin_relaxed' if relaxed else 'bin',
        datetime=to_human_datetime(start_at)
    ))
    os.makedirs(log_file_path.parent, exist_ok=True)

    result = GurobiResult(instance_id=data.id, action=Action.SCCS_BIN,
        cmd_args=sys.argv, cpu_count=os.cpu_count(), time_limit=time_limit,
        start_at=start_at, relaxed=relaxed, log_file_path=log_file_path)
    result.add_instance_data(data)

    # Init model
    init_time_start = time()
    model = build_bin_model(data, relaxed=relaxed)
    model.Params.JSONSolDetail = 1
    model.Params.LogFile = str(log_file_path.with_suffix('.gurobi.log'))
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    result.init_time = time() - init_time_start

    # Save model
    if save_model:
        model.write(str(log_file_path.with_suffix('.lp')))

    # Solve model
    def _mipsol_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            result.time_to_best = model.cbGet(GRB.Callback.RUNTIME)

    process = Process()
    start_cpu_times = process.cpu_times()
    model.optimize(_mipsol_callback)
    end_cpu_times = process.cpu_times()
    result.end_at = time()

    # Save execution info
    result.status = ResultStatus(model.status)
    result.user_time = (
        end_cpu_times.user - start_cpu_times.user +
        end_cpu_times.children_user - start_cpu_times.children_user
    )
    result.system_time = (
        end_cpu_times.system - start_cpu_times.system +
        end_cpu_times.children_system - start_cpu_times.children_system
    )
    result.solve_time = model.Runtime
    result.total_time = result.end_at - result.start_at

    # Save solution info
    result.solution_cost = model.ObjVal
    result.solution_best_bound = model.ObjBound
    result.solution_gap = model.MIPGap

    result.solution = []
    for var in model.getVars():
        if var.X != 0:
            var_name = var.VarName
            match = ONE_INDEX_SUBSET_VAR_REGEXP.match(var_name)
            if match:
                result.solution.append(int(match.group(1)))
            elif TWO_INDEX_CONFLICT_VAR_REGEXP.match(var_name):
                pass
            else:
                v0_print("[ERROR] Unexpected gurobi var name: {}", var_name)
    result.solution.sort()
    solution_info = analyse_solution(result.solution, result.solution_cost, data)
    result.add_solution_info(solution_info)

    # Save result
    model.write(str(log_file_path.with_suffix('.sol')))
    model.write(str(log_file_path.with_suffix('.json')))
    write_log(result, model)

    model.dispose()

    return result


### MIP model
def build_mip_model(data: InstanceData, relaxed=False):
    instance_id = data.id
    n_elements = data.n_elements
    subsets = data.subsets
    subset_costs = data.subset_costs
    conflict_costs = data.conflict_costs
    conflict_threshold = data.conflict_threshold

    model = gp.Model(f'{instance_id}_sccs_mip')

    SC = gp.tupledict(enumerate(subset_costs))
    X = model.addVars(SC.keys(), vtype=GRB.BINARY, name="X")

    conflict_coeff = max(c / len(s) for c,s in zip(subset_costs, subsets))
    conflict_coeff = max(round(conflict_coeff), 1)
    CC = gp.tupledict(((i,j), conflict_coeff) for i,j in conflict_costs if i < j)
    Y = model.addVars(CC.keys(), vtype=GRB.INTEGER, name="Y")

    model.setObjective(X.prod(SC) + Y.prod(CC), GRB.MINIMIZE)

    LinExpr = gp.LinExpr
    model.addConstrs(
        (LinExpr((1, X[j]) for j,s in enumerate(subsets) if i in s) >= 1
        for i in range(n_elements)), name="cov"
    )

    _addConstr = model.addConstr
    for j,t in CC.keys():
        isize = len(subsets[j].intersection(subsets[t]))
        _addConstr(isize * (X[j] + X[t] - 1) - conflict_threshold <= Y[j,t], name="conf")

    if relaxed:
        _model = model
        model = model.relax()
        _model.dispose()

    model.update()

    return model


def run_sccs_mip(data: InstanceData, time_limit=None, relaxed=False,
        save_model=False) -> GurobiResult:

    v0_print(
        f"\n{'-' * 80}\n"  "[{}] Solving '{}' SCCS INSTANCE using MIP MODEL\n"
        "(n_elements={}, n_subsets={}, conflict_threshold={}, "
        "time_limit={}, relaxed={}, save_model={})\n"  f"{'-' * 80}",
        current_human_datetime(), data.id, data.n_elements, data.n_subsets,
        data.conflict_threshold, time_limit, relaxed, save_model
    )

    start_at = time()
    log_file_path = Path(LOGS_DIR_PATH, LOG_FILE_NAME_TEMPLATE.format(
        base=os.path.splitext(data.id)[0],
        type='mip_relaxed' if relaxed else 'mip',
        datetime=to_human_datetime(start_at)
    ))
    os.makedirs(log_file_path.parent, exist_ok=True)

    result = GurobiResult(instance_id=data.id, action=Action.SCCS_MIP,
        cmd_args=sys.argv, cpu_count=os.cpu_count(), time_limit=time_limit,
        start_at=start_at, relaxed=relaxed, log_file_path=log_file_path)
    result.add_instance_data(data)

    # Init model
    init_time_start = time()
    model = build_mip_model(data, relaxed=relaxed)
    model.Params.JSONSolDetail = 1
    model.Params.LogFile = str(log_file_path.with_suffix('.gurobi.log'))
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    result.init_time = time() - init_time_start

    # Save model
    if save_model:
        model.write(str(log_file_path.with_suffix('.lp')))

    # Solve model
    def _mipsol_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            result.time_to_best = model.cbGet(GRB.Callback.RUNTIME)

    process = Process()
    start_cpu_times = process.cpu_times()
    model.optimize(_mipsol_callback)
    end_cpu_times = process.cpu_times()
    result.end_at = time()

    # Save execution info
    result.status = ResultStatus(model.status)
    result.user_time = (
        end_cpu_times.user - start_cpu_times.user +
        end_cpu_times.children_user - start_cpu_times.children_user
    )
    result.system_time = (
        end_cpu_times.system - start_cpu_times.system +
        end_cpu_times.children_system - start_cpu_times.children_system
    )
    result.solve_time = model.Runtime
    result.total_time = result.end_at - result.start_at

    # Save solution info
    result.solution_cost = model.ObjVal
    result.solution_best_bound = model.ObjBound
    result.solution_gap = model.MIPGap

    result.solution = []
    for var in model.getVars():
        if var.X != 0:
            var_name = var.VarName
            match = ONE_INDEX_SUBSET_VAR_REGEXP.match(var_name)
            if match:
                result.solution.append(int(match.group(1)))
            elif TWO_INDEX_CONFLICT_VAR_REGEXP.match(var_name):
                pass
            else:
                v0_print("[ERROR] Unexpected gurobi var name: {}", var_name)
    result.solution.sort()
    solution_info = analyse_solution(result.solution, result.solution_cost, data)
    result.add_solution_info(solution_info)

    # Save result
    model.write(str(log_file_path.with_suffix('.sol')))
    model.write(str(log_file_path.with_suffix('.json')))
    write_log(result, model)

    model.dispose()

    return result
