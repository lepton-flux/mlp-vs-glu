from abc import ABC, abstractmethod
from typing import Any
from typing import Protocol
from typing import Callable
from src.domain.metrics import Metric

class Classifier(Protocol):
    model: Callable
    criterion: Callable
    optimizer: Callable

    @property
    def id(self) -> Any:...

    def fit(self, *args, **kwargs):...

    def evaluate(self, *args, **kwargs):...

    def store(self, metric: Metric):...

class Repository(ABC):
    
    @abstractmethod
    def store(self, classifier: Classifier):...

    @abstractmethod
    def restore(self, classifier: Classifier):...