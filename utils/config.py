import os


_current_dir = os.path.dirname(os.path.abspath(__file__))


# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(_current_dir, '..'))
DATASETS_DIR_PATH = os.path.join(PROJECT_ROOT, 'datasets')
OUTPUT_DIR_PATH = os.path.join(PROJECT_ROOT, 'output')
LOGS_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, 'logs')
RESULTS_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, 'results')
ANALYSES_DIR_PATH = os.path.join(OUTPUT_DIR_PATH, 'analyses')

# File names
LOG_FILE_NAME_TEMPLATE = "{base}_sccs_{type}_{datetime}.log"

# Numeric values
FLOAT_ABS_TOLERANCE = 1e-10
