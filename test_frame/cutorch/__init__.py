__all__ = ['sgd', 'data', 'log_it']

import gpu_check

from .utils import *
from .utils.model_store import save
from .utils.model_store import load

from .optim import *

from mathstats import standardize
