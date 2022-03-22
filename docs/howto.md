# How to

## Setup
This program requires `Python 3.10`.
The only external libraries required are `gurobipy` and `psutil`, they can be installed with

```bash
pip install -r requirements.txt
```

## Usage

The following is the output of `python main.py -h` and it shows the available command line arguments.

```txt
usage: main.py [-a [ACTION ...]]
               [-c CPU_COUNT]
               [-h]
               [-k CONFLICT_THRESHOLD]
               [-r]
               [-s]
               [-t TIME_LIMIT]
               [-v]
               [--grasp-iterations-limit ITERATIONS_LIMIT]
               [--grasp-max-candidates MAX_CANDIDATES]
               [--grasp-stage1-time-limit TIME_LIMIT]
               [--grasp-stage2-time-limit TIME_LIMIT]
               [--grasp-presolve z]
               [--grasp-profile]
               [--grasp-repetitions REPETITIONS]
               [--grasp-stale-iterations-limit STALE_ITERATIONS_LIMIT]
               [--gurobi-relaxed]
               [--gurobi-save-model]
               TARGET

Common arguments:
  arguments that affect all (or most) types of actions

  TARGET                relative path to a data set file or a directory thereof
                        inside the 'datasets/' folder (e.g.: 'beasley/dataset.txt')
  -a [ACTION ...], --actions [ACTION ...]
                        the list of actions to execute for each dataset, chosen
                        among (case insensitive): ANALYSE, SCCS_BIN, SCCS_MIP,
                        SCCS_GRASP (default)
  -h, --help            show this help message and exit
  -k CONFLICT_THRESHOLD, --conflict-threshold CONFLICT_THRESHOLD
                        threshold above which two sets are considered to be in
                        conflict (default: 1)
  -c CPU_COUNT, --cpu-count CPU_COUNT
                        number of available cpu to use (default: the number of
                        logical cores available on this machine, i.e. 8).
                        When CPU_COUNT > 1 the GRASP algorithm will always use
                        at most CPU_COUNT-1 worker processes as one process is
                        reserved to handle inter process communication
  -r, --recursive       if TARGET is a directory, recursively solve all instances
                        in that directory and its subdirectories (default: false)
  -s, --silent          completely turns off output on the standard output and
                        ignores verbosity level (default: false)
  -t TIME_LIMIT, --time-limit TIME_LIMIT
                        amount of time (in seconds) after which the execution
                        is halted (default: None). When this value is not none
                        and the 2-stage GRASP algorithm is executed, each stage
                        has a time limit equal to half the value of this argument
                        (unless '--grasp-stage1-time-limit' or '--grasp-stage2-time-limit')
                        have been specified
  -v, --verbose         set the verbosity of the output on standard output.
                        Available levels: 0 (default), 1 (-v) or 2 (-vv).
                        Higher levels prints more information.

Grasp algorithm arguments:
  Arguments that affect only the SCCS_GRASP actions

  --grasp-iterations-limit ITERATIONS_LIMIT
                        execution is halted after ITERATIONS_LIMIT iterations
                        (default: None)
  --grasp-max-candidates MAX_CANDIDATES
                        number of best candidates among which one is chosen at
                        random in the first phase of the GRASP algorithm
                        (default: ceil(sqrt(N)) where N is the number of
                        available subsets)
  --grasp-stage1-time-limit TIME_LIMIT
                        amount of time (in seconds) after which the execution
                        is halted during the first stage of the GRASP algorithm
                        (default: None).
  --grasp-stage2-time-limit TIME_LIMIT
                        amount of time (in seconds) after which the execution
                        is halted during the second stage of the GRASP algorithm
                        (default: None)
  --grasp-presolve z    subsets available for the construction of a feasible
                        solution during the first phase of the GRASP algorithm
                        are limited to z * N, where N is the number of non-zero
                        variables in the solution of the relaxed LP model and
                        subsets are sorted descendingly according to
                        (var, -|reduced_cost|)
  --grasp-profile       each worker process that executes the algorithm will be
                        profiled and results will be stored in the 'output/logs/'
                        directory
  --grasp-repetitions REPETITIONS
                        number of times the GRASP algorithm is repeated (default: 1).
                        Since GRASP is not a deterministic algorithm it can be
                        useful to gather information on the outcome of multiple
                        executions.
  --grasp-stale-iterations-limit STALE_ITERATIONS_LIMIT
                        execution is halted if the incumbent solution has not
                        improved since STALE_ITERATIONS_LIMIT iterations (default: 50)

Gurobi arguments:
  Arguments that affect only SCCS_BIN and SCCS_MIP actions

  --gurobi-relaxed      the associated LP problem is solved instead of the integer
                        one when performing a SCCS_BIN or SCCS_MIP action
                        (default: false).
  --gurobi-save-model   save the model description in a file with extension '.lp'
                        (default: false). Warning: model files may be huge (>1GB)
```

## Examples

  * `python main.py beasley/scpa1-3.txt`

    Solves the problem described by the data set file located at `datasets/beasley/scpa1-3.txt` using the GRASP algorithm (the default action) with conflict threshold k=1 (the default value).

  * `python main.py beasley/`

    Solves all problems described by the data set files inside the `datasets/beasley/` folder using the GRASP algorithm (the default action).

  * `python main.py beasley/ -a SCCS_GRASP SCCS_BIN`

    Each data set file inside the folder `datasets/beasley/` is solved using the GRASP algorithm and the BIN model (the result of each action is saved in one or more text files inside `logs/beasley/`).
    *(NOTE: if `-a` is the last optional parameter before `TARGET`, the list of actions must be terminated by ` -- `.)*

  * `python main.py beasley/scpa1-3.txt -a SCCS_MIP -t 3600`

    Solves the problem described by the data set file located at `datasets/beasley/scpa1-3.txt` with conflict threshold equal to `1` (the default value) using the MIP model and stops after 3600 seconds if an optimal solution has not been found before.
