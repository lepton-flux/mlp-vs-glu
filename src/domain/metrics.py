from abc import ABC, abstractmethod
from typing import Protocol

class Metric(Protocol):
    name: str
    value: float
    phase: str
    epoch: int
    batch: int

class Metrics(ABC):

    @abstractmethod
    def build(self, **kwargs) -> Metric:...

    @abstractmethod
    def add(self, metric: Metric):...
    
    @abstractmethod
    def list(self) -> list[Metric]:...

    @abstractmethod
    def clear(self):...