from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseOperator(ABC):
    """
    A stateful operator.

    Operators receive and mutate a shared `state` dict, and return a structured
    record that will be appended to the executor trace.
    """

    name: str = "base"

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
