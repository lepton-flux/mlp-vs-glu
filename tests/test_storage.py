from torch import allclose
from torch.nn import Module
from torch.nn import Sequential, Dropout, Linear, ReLU, Flatten
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.adapters.classifier import Classifier
from src.adapters.repository import Repository

class MLP(Module):
    def __init__(self, input_features, hidden_size, output_features):
        super().__init__()
        self.layers = Sequential(
            Flatten(),
            Linear(input_features, hidden_size),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_size, output_features)
        )
    
    def forward(self, features):
        return self.layers(features)
    
def test_storing_and_restoring():
    repository = Repository('tests', 'data/test')
    repository.models.register(MLP)
    repository.optimizers.register(Adam)
    repository.criterions.register(CrossEntropyLoss)
    
    model = MLP(784, 128, 10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    classifier = Classifier(model, criterion, optimizer)
    
    repository.store(classifier)

    model = MLP(784, 128, 10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    classifier = Classifier(model, criterion, optimizer)
    other = Classifier(model, criterion, optimizer)
    repository.restore(other)
    assert allclose(classifier.model.layers[1].weight, other.model.layers[1].weight)