import gurobipy as gp


### Solver
def solve_scce(instance_id, conflict_threshold, incidence_matrix, costs, relaxed=False):
    print("\n--------------------------------------------------------------------------------")
    print(f"Solving {instance_id} with ELEMENT CONFLICTS with MIP (conflict_threshold={conflict_threshold}, relaxed={relaxed})")
    print("--------------------------------------------------------------------------------")

    n_elements = len(incidence_matrix)
    n_subsets = len(incidence_matrix[0])
    subsets = tuple(i for i in range(n_subsets))
    conflicts = tuple(i for i in range(n_elements))

    model = gp.Model("scce")

    A = incidence_matrix
    C = gp.tupledict(enumerate(costs))
    X = model.addVars(subsets, vtype=gp.GRB.BINARY if not relaxed else gp.GRB.CONTINUOUS, name="subset")
    Y = model.addVars(conflicts, vtype=gp.GRB.BINARY if not relaxed else gp.GRB.CONTINUOUS, name="conflict")

    model.setObjective(X.prod(C) + Y.sum(), gp.GRB.MINIMIZE)
    
    model.addConstrs((gp.quicksum(A[i][j] * X[j] for j in range(n_subsets)) >= 1 for i in range(n_elements)), name="coveringConstraint")
    for i in range(n_elements):
        model.addConstr(gp.quicksum(A[i][j] * X[j] for j in range(n_subsets)) - conflict_threshold <= (n_subsets - conflict_threshold) * Y[i], name="conflictConstraint1")
        model.addConstr(gp.quicksum(A[i][j] * X[j] for j in range(n_subsets)) >= (conflict_threshold + 1) * Y[i], name="conflictConstraint2")
    
    model.optimize()
    
    return model
