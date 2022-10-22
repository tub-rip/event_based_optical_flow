import logging
import os
from typing import Tuple

import h5py
import numpy as np

from ..utils import estimate_corresponding_gt_flow, undistort_events
from . import DataLoaderBase

logger = logging.getLogger(__name__)


# hdf5 data loader
def h5py_loader(path: str):
    """Basic loader for .hdf5 files.
    Args:
        path (str) ... Path to the .hdf5 file.

    Returns:
        timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
        davis_left (dict) ... "event": np.ndarray.
        davis_right (dict) ... "event": np.ndarray.
    """
    data = h5py.File(path, "r")
    event_timestamp = get_timestamp_index(data)
    r = {
        "event": np.array(data["davis"]["right"]["events"], dtype=np.int16),
    }
    # 'gray_ts': np.array(data['davis']['right']['image_raw_ts'], dtype=np.float64)
    l = {
        "event": np.array(data["davis"]["left"]["events"], dtype=np.int16),
        "gray_ts": np.array(data["davis"]["left"]["image_raw_ts"], dtype=np.float64),
    }
    data.close()
    return event_timestamp, l, r


def get_timestamp_index(h5py_data):
    """Timestampm loader for pre-fetching before actual sensor data loading.
    This is necessary for sync between sensors and decide which timestamp to
    be used as ground clock.
    Args:
        h5py_data... h5py object.
    Returns:
        timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
    """
    timestamp = {}
    timestamp["right"] = np.array(h5py_data["davis"]["right"]["events"][:, 2])
    timestamp["left"] = np.array(h5py_data["davis"]["left"]["events"][:, 2])
    return timestamp


class MvsecDataLoader(DataLoaderBase):
    """Dataloader class for MVSEC dataset."""

    NAME = "MVSEC"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    # Override
    def set_sequence(self, sequence_name: str, undistort: bool = False) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        logger.info(f"Undistort events = {undistort}")

        self.dataset_files = self.get_sequence(sequence_name)
        ts, l_event, r_event = h5py_loader(self.dataset_files["event"])
        self.left_event = l_event["event"]  # int16 .. for smaller memory consumption.
        self.left_ts = ts["left"]  # float64
        self.left_gray_ts = l_event["gray_ts"]  # float64
        # self.right_event = r_event["event"]
        # self.right_ts = ts["right"]
        # self.right_gray_ts = r_event["gray_ts"]  # float64

        # Setup GT
        if self.gt_flow_available:
            self.setup_gt_flow(os.path.join(self.gt_flow_dir, sequence_name))
            self.omit_invalid_data(sequence_name)

        # Undistort - most likely necessary to run evaluation with GT.
        self.undistort = undistort
        if self.undistort:
            self.calib_map_x, self.calib_map_y = self.get_calib_map(
                self.dataset_files["calib_map_x"], self.dataset_files["calib_map_y"]
            )

        # Setting up time suration statistics
        self.min_ts = self.left_ts.min()
        self.max_ts = self.left_ts.max()
        # self.min_ts = np.max([self.left_ts.min(), self.right_ts.min()])
        # self.max_ts = np.min([self.left_ts.max(), self.right_ts.max()]) - 10.0  # not use last 1 sec
        self.data_duration = self.max_ts - self.min_ts

    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `outdoot_day2`.

        Returns:
            sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        data_path: str = os.path.join(self.root_dir, sequence_name)
        event_file = data_path + "_data.hdf5"
        calib_file_x = data_path[:-1] + "_left_x_map.txt"
        calib_file_y = data_path[:-1] + "_left_y_map.txt"
        sequence_file = {
            "event": event_file,
            "calib_map_x": calib_file_x,
            "calib_map_y": calib_file_y,
        }
        return sequence_file

    def setup_gt_flow(self, path):
        path = path + "_gt_flow_dist.npz"
        logger.info(f"Loading ground truth flow {path}")
        gt = np.load(path)
        self.gt_timestamps = gt["timestamps"]
        self.U_gt_all = gt["x_flow_dist"]
        self.V_gt_all = gt["y_flow_dist"]

    def free_up_flow(self):
        del self.gt_timestamps, self.U_gt_all, self.V_gt_all

    def omit_invalid_data(self, sequence_name: str):
        logger.info(f"Use only valid frames.")
        first_valid_gt_frame = 0
        last_valid_gt_frame = -1
        if "indoor_flying1" in sequence_name:
            first_valid_gt_frame = 60
            last_valid_gt_frame = 1340
        elif "indoor_flying2" in sequence_name:
            first_valid_gt_frame = 140
            last_valid_gt_frame = 1500
        elif "indoor_flying3" in sequence_name:
            first_valid_gt_frame = 100
            last_valid_gt_frame = 1711
        elif "indoor_flying4" in sequence_name:
            first_valid_gt_frame = 104
            last_valid_gt_frame = 380
        elif "outdoor_day1" in sequence_name:
            last_valid_gt_frame = 5020
        elif "outdoor_day2" in sequence_name:
            first_valid_gt_frame = 30
            # last_valid_gt_frame = 5020

        self.gt_timestamps = self.gt_timestamps[first_valid_gt_frame:last_valid_gt_frame]
        self.U_gt_all = self.U_gt_all[first_valid_gt_frame:last_valid_gt_frame]
        self.V_gt_all = self.V_gt_all[first_valid_gt_frame:last_valid_gt_frame]

        # Update event list
        first_event_index = self.time_to_index(self.gt_timestamps[0])
        last_event_index = self.time_to_index(self.gt_timestamps[-1])
        self.left_event = self.left_event[first_event_index:last_event_index]
        self.left_ts = self.left_ts[first_event_index:last_event_index]

        self.min_ts = self.left_ts.min()
        self.max_ts = self.left_ts.max()

        # Update gray frame ts
        self.left_gray_ts = self.left_gray_ts[
            (self.gt_timestamps[0] < self.left_gray_ts)
            & (self.gt_timestamps[-1] > self.left_gray_ts)
        ]

        # self.right_event = self.right_event[first_event_index:last_event_index]
        # self.right_ts = self.right_ts[first_event_index:last_event_index]
        # self.right_gray_ts = self.right_gray_ts[
        #     (self.gt_timestamps[0] < self.right_gray_ts)
        #     & (self.gt_timestamps[-1] > self.right_gray_ts)
        # ]

    def __len__(self):
        return len(self.left_event)

    def load_event(self, start_index: int, end_index: int, cam: str = "left") -> np.ndarray:
        """Load events.
        The original hdf5 file contains (x, y, t, p),
        where x means in width direction, and y means in height direction. p is -1 or 1.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value, in sec. p is [-1, 1].
        """
        n_events = end_index - start_index
        events = np.zeros((n_events, 4), dtype=np.float64)

        if cam == "left":
            if len(self.left_event) <= start_index:
                logger.error(
                    f"Specified {start_index} to {end_index} index for {len(self.left_event)}."
                )
                raise IndexError
            events[:, 0] = self.left_event[start_index:end_index, 1]
            events[:, 1] = self.left_event[start_index:end_index, 0]
            events[:, 2] = self.left_ts[start_index:end_index]
            events[:, 3] = self.left_event[start_index:end_index, 3]
        elif cam == "right":
            logger.error("Please select `left`as `cam` parameter.")
            raise NotImplementedError
        if self.undistort:
            events = undistort_events(
                events, self.calib_map_x, self.calib_map_y, self._HEIGHT, self._WIDTH
            )
        return events

    # Optical flow (GT)
    def gt_time_list(self):
        return self.gt_timestamps

    def eval_frame_time_list(self):
        # In MVSEC, evaluation is based on gray frame timestamp.
        return self.left_gray_ts

    def index_to_time(self, index: int) -> float:
        return self.left_ts[index]

    def time_to_index(self, time: float) -> int:
        # inds = np.where(self.left_ts > time)[0]
        # if len(inds) == 0:
        #     return len(self.left_ts) - 1
        # return inds[0] - 1
        ind = np.searchsorted(self.left_ts, time)
        return ind - 1

    def get_gt_time(self, index: int) -> tuple:
        """Get GT flow timestamp [floor, ceil] for a given index.

        Args:
            index (int): Index of the event

        Returns:
            tuple: [floor_gt, ceil_gt]. Both are synced with GT optical flow.
        """
        inds = np.where(self.gt_timestamps > self.index_to_time(index))[0]
        if len(inds) == 0:
            return (self.gt_timestamps[-1], None)
        elif len(inds) == len(self.gt_timestamps):
            return (None, self.gt_timestamps[0])
        else:
            return (self.gt_timestamps[inds[0] - 1], self.gt_timestamps[inds[0]])

    def load_optical_flow(self, t1: float, t2: float) -> np.ndarray:
        """Load GT Optical flow based on timestamp.
        Note: this is pixel displacement.
        Note: the args are not indices, but timestamps.

        Args:
            t1 (float): [description]
            t2 (float): [description]

        Returns:
            [np.ndarray]: H x W x 2. Be careful that the 2 ch is [height, width] direction component.
        """
        U_gt, V_gt = estimate_corresponding_gt_flow(
            self.U_gt_all,
            self.V_gt_all,
            self.gt_timestamps,
            t1,
            t2,
        )
        gt_flow = np.stack((V_gt, U_gt), axis=2)
        return gt_flow

    def load_calib(self) -> dict:
        """Load calibration file.

        Outputs:
            (dict) ... {"K": camera_matrix, "D": distortion_coeff}
                camera_matrix (np.ndarray) ... [3 x 3] matrix.
                distortion_coeff (np.array) ... [5] array.
        """
        logger.warning("directly load calib_param is not implemented!! please use rectify instead.")
        outdoor_K = np.array(
            [
                [223.9940010790056, 0, 170.7684322973841, 0],
                [0, 223.61783486959376, 128.18711828338436, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        return {"K": outdoor_K}

    def get_calib_map(self, map_txt_x, map_txt_y):
        """Intrinsic calibration parameter file loader.
        Args:
            map_txt... file path.
        Returns
            map_array (np.array)... map array.
        """
        map_x = self.load_map_txt(map_txt_x)
        map_y = self.load_map_txt(map_txt_y)
        return map_x, map_y

    def load_map_txt(self, map_txt):
        f = open(map_txt, "r")
        line = f.readlines()
        map_array = np.zeros((self._HEIGHT, self._WIDTH))
        for i, l in enumerate(line):
            map_array[i] = np.array([float(k) for k in l.split()])
        return map_array
