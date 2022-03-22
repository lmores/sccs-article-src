import os
import time

import numpy as np
import gurobipy as gp

import utils
from . import utils

### Solver
def solve_ssc(instance_id, incidence_matrix, time_limit=None, relaxed=False):
    instance_id = instance_id.removeprefix(f'datasets{os.path.sep}')
    print("\n--------------------------------------------------------------------------------")
    print(f"Solving {instance_id} for SIMPLE set covering with MIP (relaxed={relaxed})")
    print("--------------------------------------------------------------------------------")

    base_file_path = os.path.join(utils.misc.LOGS_DIR_PATH, os.path.splitext(instance_id)[0])
    file_path_noext = f"{base_file_path}_ssc_{'lp' if relaxed else 'mip'}_{utils.misc.current_human_datetime()}"

    result = utils.datamodels.GurobiResult(instance_id,
        len(incidence_matrix), len(incidence_matrix[0]), None, None, time_limit, relaxed)

    result.log_file_path = f'{file_path_noext}.log'
    utils.log_instance_info(result)

    # Init model
    init_start_time = time.time()
    model = gp.Model("ssc")

    model.setParam(gp.GRB.Param.LogFile, result.log_file_path)
    if time_limit is not None:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    A = np.array(incidence_matrix)
    X = model.addMVar(shape=len(incidence_matrix[0]), vtype=gp.GRB.BINARY, name="subset")

    model.setObjective(sum(X), gp.GRB.MINIMIZE)
    model.addConstr(A @ X >= 1, name="coveringConstraint")

    result.init_time = time.time() - init_start_time

    # Write model
    model.Params.JSONSolDetail = 1
    for var in model.getVars():
        var.VTag = var.VarName
    for constr in model.getConstrs():
        constr.CTag = constr.ConstrName
    model.write(os.path.join(utils.misc.LOGS_DIR_PATH, f'{file_path_noext}.lp'))

    # Solve model
    solve_start_time = time.time()
    model.optimize()
    result.solve_time = time.time() - solve_start_time
    result.status = model.status

    # Write solutions
    model.write(os.path.join(utils.misc.LOGS_DIR_PATH, f'{file_path_noext}.sol'))
    model.write(os.path.join(utils.misc.LOGS_DIR_PATH, f'{file_path_noext}.json'))
    utils.log_solution_info(model, result)

    return model, result


### Runner
def run_ssc(instance_id, incidence_matrix, time_limit=None, relaxed=False) -> utils.datamodels.GurobiResult:
    model, result = solve_ssc(instance_id, incidence_matrix, time_limit=time_limit, relaxed=relaxed)
    result.solution_cost = model.getObjective().getValue()
    sol_var_names = set(var.getAttr(gp.GRB.Attr.VarName) for var in model.getVars() if var.getAttr(gp.GRB.Attr.X) != 0)
    result.solution = sorted(int(utils.ONE_INDEX_SUBSET_VAR_REGEXP.match(name).group(1)) for name in sol_var_names)
    model.dispose()

    return result
