from os import path
from typing import Optional
from typing import Callable
from logging import getLogger
from mlregistry import Registry
from mlregistry import get_hash, get_metadata, get_date_hash
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torchsystem.weights import Weights

logger = getLogger(__name__)

class Storage[T]:
    weights: Optional[Weights[T]]
    registry: Registry[T]
    category: str
    
    @classmethod
    def register(cls, type: type):
        cls.registry.register(type, cls.category)
        return type

    @classmethod
    def build(cls, name: str, *args, **kwargs) -> Optional[T]:
        '''
        Build an object from the registry.

        Args:
            name (str): The name of the object.
            *args: Positional arguments for initializing the object.
            **kwargs: Keyword arguments for initializing the object.

        Returns:
            Optional[T]: The object from the registry.
        '''
        if not name in cls.registry.keys():
            return None
        object = cls.registry.get(name)(*args, **kwargs)
        return object
        
    def get(self, name: str, *args, **kwargs) -> Optional[T]:
        """
        Get an object from the registry and restore its weights.

        Args:
            name (str): The name of the object.
            *args: Positional arguments for initializing the object.
            **kwargs: Keyword arguments for initializing the object.

        Returns:
            Optional[T]: The object from the registry with restored weights if available.
        """
        if not name in self.registry.keys():
            return None
        object = self.registry.get(name)(*args, **kwargs)
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{get_hash(object)}' )
        return object
    
    def store(self, object: T):
        """
        Store the object's weights if available.

        Args:
            object (T): The object to store its weights.
        """
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{get_hash(object)}' )

    def restore(self, object: T):
        """
        Restore the object's weights if available.

        Args:
            object (T): The object to restore its weights.
        """
        assert object.__class__.__name__ in self.registry.keys(), f'{object.__class__.__name__} not registered in {self.category}'
        if hasattr(self, 'weights'):
            print(f'{self.category}-{object.__class__.__name__}-{get_hash(object)}')
            self.weights.restore(object, f'{self.category}-{object.__class__.__name__}-{get_hash(object)}' )


class Models(Storage[Module]):
    category = 'model'
    registry = Registry()

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Criterions(Storage[Module]):
    category = 'criterion'
    registry = Registry()

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Optimizers(Storage[Optimizer]):
    category = 'optimizer'
    registry = Registry(excluded_positions=[0], exclude_parameters={'params'})

    def __init__(
        self, 
        root: str = 'data/weights', 
        path: str = None,
        extension: str = '.pth'
    ):
        self.weights = Weights(root, path, extension)


class Datasets(Storage[Dataset]):
    category = 'dataset'
    registry = Registry(exclude_parameters={'root', 'download'})
