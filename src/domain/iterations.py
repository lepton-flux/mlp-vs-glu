from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime
from typing import Protocol

class Dataset(Protocol):
    hash: str
    name: str
    arguments: dict[str, Any]

class Loader(Protocol):
    dataset: str
    arguments: dict[str, Any]
    
class Iteration(Protocol):
    hash: str
    epochs: int
    loaders: list[tuple[str, Loader]]
    start: datetime
    end: datetime

class Iterations(ABC):
    
    @abstractmethod
    def list(self) -> list[Iteration]:...
    
    @abstractmethod
    def put(self, iteration: Iteration):...
    
    @abstractmethod
    def clear(self):...