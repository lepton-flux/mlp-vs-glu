from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from mlregistry import get_hash

class Classifier:
    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.epoch = 0
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    @property
    def id(self) -> str:
        return get_hash(self.model)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
    
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)

    def fit(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]:
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def evaluate(self, input: Tensor, target: Tensor) -> tuple[Tensor, float]: 
        output = self(input)
        loss = self.loss(output, target)
        return output, loss.item()
