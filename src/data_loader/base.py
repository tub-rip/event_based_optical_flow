import logging
import os

import numpy as np

from .. import utils
from . import DATASET_ROOT_DIR

logger = logging.getLogger(__name__)


class DataLoaderBase(object):
    """Base of the DataLoader class.
    Please make sure to implement
     - load_event()
     - get_sequence()
    in chile classes.
    """

    NAME = "example"

    def __init__(self, config: dict = {}):
        self._HEIGHT = config["height"]
        self._WIDTH = config["width"]

        root_dir: str = config["root"] if config["root"] else DATASET_ROOT_DIR
        self.root_dir: str = os.path.expanduser(root_dir)
        data_dir: str = config["dataset"] if config["dataset"] else self.NAME

        self.dataset_dir: str = os.path.join(self.root_dir, data_dir)
        self.__dataset_files: dict = {}
        logger.info(f"Loading directory in {self.dataset_dir}")

        self.gt_flow_available: bool
        if utils.check_key_and_bool(config, "load_gt_flow"):
            self.gt_flow_dir: str = os.path.expanduser(config["gt"])
            self.gt_flow_available = utils.check_file_utils(self.gt_flow_dir)
        else:
            self.gt_flow_available = False

        if utils.check_key_and_bool(config, "undistort"):
            logger.info("Undistort events when load_event.")
            self.auto_undistort = True
        else:
            logger.info("No undistortion.")
            self.auto_undistort = False

    @property
    def dataset_files(self) -> dict:
        return self.__dataset_files

    @dataset_files.setter
    def dataset_files(self, sequence: dict):
        self.__dataset_files = sequence

    def set_sequence(self, sequence_name: str) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        self.dataset_files = self.get_sequence(sequence_name)

    def get_sequence(self, sequence_name: str) -> dict:
        raise NotImplementedError

    def load_event(
        self, start_index: int, end_index: int, cam: str = "left", *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def load_calib(self) -> dict:
        raise NotImplementedError

    def load_optical_flow(self, t1: float, t2: float, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def index_to_time(self, index: int) -> float:
        raise NotImplementedError

    def time_to_index(self, time: float) -> int:
        raise NotImplementedError
