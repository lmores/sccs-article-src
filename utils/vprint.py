from typing import Any, Mapping


_VERBOSITY_LEVEL = (0, False)    # (level_value, silent)


# Implementation must be done in separate internal functions
# in order to make monkey patching work with named imports
def _vprint_impl(msg: str, *args, msg_kwargs: Mapping[str,Any] = {},
        end='\n', flush=False) -> None:
    text = msg.format(*args, **msg_kwargs) if args or msg_kwargs else msg
    print(text, end=end, flush=flush)

def _v0_print(*args, **kwargs) -> None:
    pass

def _v1_print(*args, **kwargs) -> None:
    pass

def _v2_print(*args, **kwargs) -> None:
    pass

def _vprint_noop(*args, **kwargs) -> None:
    pass


# Public functions
def get_verbosity_level() -> tuple[int, bool]:
    return _VERBOSITY_LEVEL

def set_verbosity_level(level=0, silent=False) -> None:
    global _VERBOSITY_LEVEL
    _VERBOSITY_LEVEL = (level, silent)

    global _v0_print, _v1_print, _v2_print
    _v0_print = _vprint_impl if not silent and level >= 0 else _vprint_noop
    _v1_print = _vprint_impl if not silent and level >= 1 else _vprint_noop
    _v2_print = _vprint_impl if not silent and level >= 2 else _vprint_noop

def v0_print(msg: str, *args, msg_kwargs: Mapping[str, Any] = {},
        end='\n', flush=False) -> None:
    _v0_print(msg, *args, msg_kwargs=msg_kwargs, end=end, flush=flush)

def v1_print(msg: str, *args, msg_kwargs: Mapping[str, Any] = {},
        end='\n', flush=False) -> None:
    _v1_print(msg, *args, msg_kwargs=msg_kwargs, end=end, flush=flush)

def v2_print(msg: str, *args, msg_kwargs: Mapping[str, Any] = {},
        end='\n', flush=False) -> None:
    _v2_print(msg, *args, msg_kwargs=msg_kwargs, end=end, flush=flush)
