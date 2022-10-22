import os

DATASET_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets"
)

from .base import DataLoaderBase

# from .davis_data import DavisDataLoader
# from .dsec import DsecDataLoader   # TODO comes later..
from .mvsec import MvsecDataLoader


# List of supported dataset
def inheritors(klass):
    subclasses = set()
    work = [klass]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


collections = {k.NAME: k for k in inheritors(DataLoaderBase)}

# DNN
# TODO comes lates
