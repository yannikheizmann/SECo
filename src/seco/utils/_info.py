from __future__ import annotations
from typing import Any



class ExecutionTime:
    """
    Singleton class that tracks timing metrics across various stages and agents involved in the application.
    This includes start/end times and average execution times for multiple agents like sweaty, hso, general, etc.
    """
    _instance: 'ExecutionTime' = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExecutionTime, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return
        self.__class__._initialized = True
        # Set all default values
        self.et_overall_start = 0.0
        self.et_overall_end = 0.0
        self.et_intent = 0.0
        self.et_sweaty_avg = (0, 0.0)
        self.et_sweaty_start = 0.0
        self.et_sweaty_end = 0.0
        self.et_hso_avg = (0, 0.0)
        self.et_hso_start = 0.0
        self.et_hso_end = 0.0
        self.et_general_avg = (0, 0.0)
        self.et_general_start = 0.0
        self.et_general_end = 0.0
        self.et_aggregator = 0.0

    @classmethod
    def reset(cls) -> None:
        """
        Resets the singleton instance's attributes to their default values.
        """
        cls._instance = None
        cls._initialized = False
        cls()  # force reinitialization

    @classmethod
    def update(cls, timing_data: dict[str, Any]) -> None:
        """
        Updates the execution time metrics in the singleton instance with new timing data.

        Args:
            timing_data (dict): Dictionary containing timing keys and their corresponding values.
                                Only existing attributes in the instance will be updated.
        """
        #print("updating,", timing_data)
        instance = cls()
        for key, value in timing_data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
                #print(f"Updated {key} to {value}")
        #print(ExecutionTime.to_pretty_string())

    @classmethod
    def to_pretty_string(cls) -> str:
        """
        Returns a nicely formatted string of the current execution time values.
        """
        instance = cls()
        output = ["ExecutionTime Summary:"]
        for attr in dir(instance):
            if attr.startswith("et_") and not callable(getattr(instance, attr)) and not attr.startswith("__"):
                output.append(f"  {attr}: {getattr(instance, attr)}")
        return "\n".join(output)