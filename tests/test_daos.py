from datetime import datetime
from src.adapters.metrics import Metrics, Metric
from src.adapters.modules import Modules, Module
from src.adapters.iterations import Dataset, Loader, Iteration, Iterations

def test_metrics(metrics: Metrics):
    metrics.add(Metric(name='accuracy', value=0.9, batch=100, epoch=1, phase='train'))
    metrics.add(Metric(name='accuracy', value=0.2, batch=100, epoch=2, phase='train'))
    metrics.add(Metric(name='accuracy', value=0.2, batch=100, epoch=2, phase='test'))
    metrics.add(Metric(name='loss', value=0.3, batch=100, epoch=1, phase='train'))
    assert len(metrics.list()) == 4
    metrics.clear()
    assert len(metrics.list()) == 0

    metric = metrics.build(name='accuracy', value=0.9, batch=100, epoch=1, phase='train')
    assert metric.owner is not None

def test_modules(modules: Modules):
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=1, arguments={'lr': 0.1}))
    modules.put(Module(type='criterion', hash='hash1', name='CrossEntropyLoss', epoch=1, arguments={}))
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=10, arguments={'lr': 0.1}))
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=11, arguments={'lr': 0.1})) #UPDATE
    assert len(modules.list('optimizer')) == 1

    modules.put(Module(type='optimizer', hash='hash2', name='Adam', epoch=12, arguments={'lr': 0.2})) #ADD SINCE LAST HASH IS DIFFERENT
    assert len(modules.list('optimizer')) == 2

    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=13, arguments={'lr': 0.1})) #ADD SINCE LAST HASH IS DIFFERENT
    modules.put(Module(type='optimizer', hash='hash1', name='Adam', epoch=14, arguments={'lr': 0.1})) #UPDATE
    assert len(modules.list('optimizer')) == 3
    assert len(modules.list('criterion')) == 1

    print(modules.list('optimizer'))
    print(modules.list('criterion'))

    modules.clear()
    assert len(modules.list('optimizer')) == 0
    assert len(modules.list('criterion')) == 0

    module = modules.build(type='optimizer', hash='hash1', name='Adam', epoch=1, arguments={'lr': 0.1})
    assert module.owner is not None

def test_iterations(iterations: Iterations):
    iterations.put(Iteration.model_validate({
        'hash': "1234",
        'epochs': 5,
        'loaders': [
            ('train', {
                'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
                'arguments': {'batch_size': 64}
            }),
            ('test', {
                'dataset': {'hash': "1235", 'name': "MNIST", 'arguments': {'test': True}},
                'arguments': {'batch_size': 64}
            })
        ],
        'start': datetime.now(),
        'end': datetime.now()
    }))


    iterations.put(Iteration.model_validate({
        'hash': "1234",
        'epochs': 10,
        'loaders': [
            ('train', {
                'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
                'arguments': {'batch_size': 64}
            }),
            ('test', {
                'dataset': {'hash': "1235", 'name': "MNIST", 'arguments': {'test': True}},
                'arguments': {'batch_size': 64}
            })
        ],
        'start': datetime.now(),
        'end': datetime.now()
    }))

    assert len(iterations.list()) == 1
    
    iterations.put(Iteration.model_validate({
        'hash': "12345",
        'epochs': 10,
        'loaders': [
            ('train', {
                'dataset': {'hash': "1234", 'name': "MNIST", 'arguments': {'train': True}},
                'arguments': {'batch_size': 64}
            }),
            ('test', {
                'dataset': {'hash': "1235", 'name': "MNIST", 'arguments': {'test': True}},
                'arguments': {'batch_size': 64}
            })
        ],
        'start': datetime.now(),
        'end': datetime.now()
    }))

    assert len(iterations.list()) == 2

    assert isinstance(iterations.list()[0], Iteration)