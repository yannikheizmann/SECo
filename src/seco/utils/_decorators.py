import time
from functools import wraps
from typing import Callable, Any, Dict, Optional

from ._info import ExecutionTime


def skip_registry(cls):
    """
    Class decorator to mark a class to be skipped during registry processing.

    Adds a `__skip_registry__` attribute to the class to indicate
    that it should not be automatically registered in any registry systems.

    Args:
        cls (type): The class to decorate.

    Returns:
        type: The same class with the `__skip_registry__` attribute set to True.
    """
    cls.__skip_registry__ = True
    return cls


def time_execution(key: str, start_key: Optional[str] = None, end_key: Optional[str] = None):
    """
    Decorator factory that measures the execution time of the decorated method.

    Measures elapsed time for a method call, and inserts timing information
    into the returned Command or dict object under specified keys.

    Args:
        key (str): The key under which to store the elapsed time (in seconds).
        start_key (Optional[str]): Optional key to store the start timestamp.
        end_key (Optional[str]): Optional key to store the end timestamp.

    Returns:
        Callable: A decorator that wraps the target function, measures timing,
                  and inserts timing info into the returned Command or dict.
    """
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            result: Dict[str, Any] = fn(self, *args, **kwargs)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) 
            timing_data = {key: elapsed_ms}

            if start_key:
                timing_data[start_key] = t0   
            if end_key:
                timing_data[end_key] = t1

            if key in ["et_general_avg", "et_hso_avg", "et_sweaty_avg"]:
                et_instance = ExecutionTime()
                if hasattr(et_instance, key):
                    current_avg = getattr(et_instance, key)
                    timing_data[key] = (current_avg[0] +1, current_avg[1] + (elapsed_ms / (current_avg[0] +1)))

            elif start_key in ["et_general_start", "et_hso_start", "et_sweaty_start"]:
                if hasattr(ExecutionTime, key):
                    current_start = getattr(ExecutionTime, key)
                    if current_start != 0.0:
                        timing_data.pop(start_key) 
            ExecutionTime.update(timing_data)
            return result 
        return wrapper
    return decorator