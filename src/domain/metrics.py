from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Metric:
    name: str
    value: float
    phase: str
    epoch: int
    batch: int

class Metrics(ABC):

    @abstractmethod
    def add(self, metric: Metric):...
    
    @abstractmethod
    def list(self) -> list[Metric]:...

    @abstractmethod
    def clear(self):...