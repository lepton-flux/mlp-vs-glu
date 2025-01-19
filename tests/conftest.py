from os import path, makedirs
from shutil import rmtree
from pytest import fixture
from logging import getLogger
from tinydb import TinyDB
from src.adapters.metrics import Metrics
from src.adapters.modules import Modules
from src.adapters.iterations import Iterations

logger = getLogger(__name__)

@fixture(scope='session')
def database():
    if not path.exists('data/test'):
        makedirs('data/test')
    yield TinyDB('data/test/database.json')
    try:
        rmtree('data/test')
    except PermissionError:
        logger.warning('Could not remove data directory')


@fixture(scope='function')
def metrics(database):
    return Metrics(database, 'test_model')

@fixture(scope='function')
def modules(database):
    return Modules(database, 'test_model')

@fixture(scope='function')
def iterations(database):
    return Iterations(database, 'test_model')


