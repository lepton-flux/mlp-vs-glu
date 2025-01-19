from tinydb import TinyDB, where
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class Metric:
    name: str
    value: float
    phase: str
    epoch: int
    batch: int

class Metrics:
    def __init__(self, database: TinyDB, model: str):
        self.model = str(model)
        self.database = database
        self.table = self.database.table('metrics')
    
    def add(self, metric: Metric):
        self.table.insert({'model': self.model, **asdict(metric)})

    def list(self) -> list[Metric]:
        results = self.table.search(where('model') == self.model)
        return [Metric(**{key: value for key, value in result.items() if key != 'model'}) for result in results]
    
    def clear(self):
        self.table.remove(where('model') == self.model)