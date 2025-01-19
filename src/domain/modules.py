from abc import ABC, abstractmethod
from typing import Any
from typing import Optional
from typing import Protocol

class Module(Protocol):
    type: str
    hash: str
    name: str
    epoch: int
    arguments: dict[str, Any]

class Modules(ABC):

    @abstractmethod
    def build(self, **kwargs) -> Module:...
    
    @abstractmethod
    def list(self, type: str) -> list[Module]:...

    @abstractmethod
    def last(self, type: str) -> Optional[Module]:...
    
    @abstractmethod
    def put(self, module: Module):...