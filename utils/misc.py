import re

from datetime import datetime
from math import ceil
from random import SystemRandom
from typing import Counter, Dict, Iterable, Mapping, Tuple, Union


# Random generator
SRG = SystemRandom()


# Helpers
_INTEGER_REGEXP = re.compile('(\d+)')

def nat_sort_key(key) -> Tuple[Union[int, str], ...]:
    tokens = _INTEGER_REGEXP.split(key)
    return tuple(int(txt) if txt.isnumeric() else txt for txt in tokens)


# Dates
DATETIME_FMT = '%Y-%m-%d-%H-%M-%S-%f'

def current_human_datetime() -> str:
    return datetime.now().strftime(DATETIME_FMT)

def to_human_datetime(timestamp) -> str:
    if timestamp is None:
        return None

    return datetime.fromtimestamp(timestamp).strftime(DATETIME_FMT)


# Collection functions
def build_distrib(values: Iterable[float]) -> Dict[float,int]:
    return dict(sorted(Counter(values).items()))


def aggregate_distrib(distrib: Mapping[int,int],
        steps=3, min_step_size=1) -> Dict[Tuple[int,int], int]:

    # Note: this function is written for integer values only
    if not distrib:
        return {}

    min_value, max_value = min(distrib), max(distrib)
    interval_size = max_value - min_value + 1
    if min_step_size * steps > interval_size:
        step_size = min_step_size
        steps = ceil(interval_size / step_size)
    else:
        step_size = interval_size / steps

    aggregated_distrib = {}
    c = min_value
    while c < max_value:
        aggregated_distrib[(round(c), round(c + step_size))] = 0
        c += step_size

    for value, count in distrib.items():
        step_number = (value - min_value) // step_size
        step_start = min_value + step_size * step_number
        int_step_start = round(step_start)
        int_step_end = round(step_start + step_size)
        if value < int_step_start:
            step = (round(step_start - step_size), int_step_start)
        elif value >= int_step_end:
            step = (int_step_end, round(step_start + 2 * step_size))
        else:
            step = (int_step_start, int_step_end)
        aggregated_distrib[step] = count + aggregated_distrib.get(step, 0)

    assert sum(aggregated_distrib.values()) == sum(distrib.values())

    return aggregated_distrib