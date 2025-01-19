from torchsystem.storage import Models, Criterions, Optimizers, Datasets
from torchsystem.storage import get_metadata

from src.adapters.classifier import Classifier
from src.adapters.modules import Modules, Module

def get_module(module, epoch) -> Module:
    metadata = get_metadata(module)
    return Module(
        type=metadata.type,
        hash=metadata.hash,
        name=metadata.name,
        arguments=metadata.arguments,
        epoch=epoch
    )

class Repository:
    
    def __init__(self, experiment, database):
        self.models = Models(folder=experiment)
        self.criterions = Criterions(folder=experiment)
        self.optimizers = Optimizers(folder=experiment)
        self.datasets = Datasets()
        self.database = database

    def store(self, classifier: Classifier):
        modules = Modules(self.database, classifier.id)
        modules.put(get_module(classifier.model, classifier.epoch))
        modules.put(get_module(classifier.criterion, classifier.epoch))
        modules.put(get_module(classifier.optimizer, classifier.epoch))
        self.models.store(classifier.model)

    def restore(self, classifier: Classifier):
        modules = Modules(self.database, classifier.id)
        classifier.epoch = modules.last('model').epoch
        self.models.restore(classifier.model)