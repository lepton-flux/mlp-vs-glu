from abc import ABC, abstractmethod
from typing import Any
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from tinydb import TinyDB, where

class Dataset(BaseModel):
    hash: str
    name: str
    arguments: dict[str, Any]

class Loader(BaseModel):
    dataset: Dataset
    arguments: dict[str, Any]
    
class Schema(BaseModel):
    owner: Optional[str] = Field(default=None, repr=False)

class Iteration(Schema):
    hash: str
    epochs: int
    loaders: list[tuple[str, Loader]]
    start: datetime = Field(default=datetime.now(tz=timezone.utc))
    end: datetime = Field(default=datetime.now(tz=timezone.utc))

class Iterations(ABC):
    def __init__(self, database: TinyDB, owner: str):
        self.owner = owner
        self.database = database
        self.table = self.database.table('iterations')

    
    def put(self, iteration: Iteration):
        iteration.owner = self.owner
        iterations = self.table.search(where('owner') == self.owner)
        if not iterations:
            self.table.insert(iteration.model_dump(mode='json'))
        elif iteration.hash == iterations[-1]['hash']:
            self.table.update(iteration.model_dump(mode='json'), doc_ids=[iterations[-1].doc_id])
        else:
            self.table.insert(iteration.model_dump(mode='json'))

    def list(self) -> list[Iteration]:
        iterations = self.table.search(where('owner') == self.owner)
        return [Iteration.model_validate(iteration) for iteration in iterations]
    
    def clear(self):
        self.table.remove(where('owner') == self.owner)