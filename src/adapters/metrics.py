from tinydb import TinyDB, where
from pydantic import BaseModel, Field
from typing import Optional

class Schema(BaseModel):
    owner: Optional[str] = Field(default=None, repr=False)

class Metric(Schema):
    name: str
    value: float
    phase: str
    epoch: int
    batch: int

class Metrics:
    def __init__(self, database: TinyDB, owner: str):
        self.owner = str(owner)
        self.database = database
        self.table = self.database.table('metrics')

    def build(self, **kwargs) -> Metric:
        return Metric(**kwargs | {'owner': self.owner})
    
    def add(self, metric: Metric):
        metric.owner = self.owner
        self.table.insert(metric.model_dump())

    def list(self) -> list[Metric]:
        results = self.table.search(where('owner') == self.owner)
        return [Metric.model_validate(result) for result in results]
    
    def clear(self):
        self.table.remove(where('owner') == self.owner)