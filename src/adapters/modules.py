from typing import Any
from typing import Optional
from tinydb import TinyDB, where
from pydantic import BaseModel, Field
from typing import Optional

class Schema(BaseModel):
    owner: Optional[str] = Field(default=None, repr=False)

class Module(Schema):
    type: str
    hash: str
    name: str
    epoch: int
    arguments: dict[str, Any]

class Modules:
    def __init__(self, database: TinyDB, owner: str):
        self.owner = owner
        self.database = database
        self.table = self.database.table('modules')
    
    def build(self, **kwargs) -> Module:
        return Module(**kwargs | {'owner': self.owner})
    
    def list(self, type: str) -> list[Module]:
        return [Module.model_validate(module) for module in self.table.search(where('owner') == self.owner and where('type') == type)]
    
    def last(self, type: str) -> Optional[Module]:
        modules = self.table.search(where('owner') == self.owner and where('type') == type)
        return Module(**modules[-1]) if modules else None
    
    def put(self, module: Module):
        module.owner = self.owner
        modules = self.table.search(where('owner') == self.owner and where('type') == module.type)
        if not modules:
            self.table.insert(module.model_dump())
        elif module.hash == modules[-1]['hash']:
            self.table.update(module.model_dump(), doc_ids=[modules[-1].doc_id])
        else:
            self.table.insert(module.model_dump())

    def clear(self):
        self.table.remove(where('owner') == self.owner)