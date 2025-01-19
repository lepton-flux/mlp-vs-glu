import os
from logging import getLogger
from typing import Optional
from torch import save, load
from torch.nn import Module

logger = getLogger(__name__)

class Weights[T: Module]:

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.location = os.path.join(root, path) if path else root
        self.extension = extension
        if not os.path.exists(self.location):
            os.makedirs(self.location)
    
    def store(self, module: T, filename: Optional[str] = None):
        filename = filename or module.__class__.__name__
        save(module.state_dict(), os.path.join(self.location, filename + self.extension))

    def restore(self, module: T, filename: Optional[str]) -> bool:
        filename = filename or module.__class__.__name__
        try:
            state_dict = load(os.path.join(self.location, filename + self.extension), weights_only=True)
            module.load_state_dict(state_dict)
            return True
        except FileNotFoundError:
            return False