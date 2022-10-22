import ast
import cProfile
import logging
import os
import pstats
import random
import subprocess
from functools import wraps
from typing import Dict

import numpy as np
import optuna
import torch

logger = logging.getLogger(__name__)


def fix_random_seed(seed=46) -> None:
    """Fix random seed"""
    logger.info("Fix random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_file_utils(filename: str) -> bool:
    """Return True if the file exists.

    Args:
        filename (str): _description_

    Returns:
        bool: _description_
    """
    logger.debug(f"Check {filename}")
    res = os.path.exists(filename)
    if not res:
        logger.warning(f"{filename} does not exist!")
    return res


def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]


def fetch_runtime_information() -> dict:
    """Fetch information of the experiment at runtime.

    Returns:
        dict: _description_
    """
    config = {}
    config["commit"] = fetch_commit_id()
    config["server"] = get_server_name()
    return config


def fetch_commit_id() -> str:
    """Get the latest commit ID of the repository.

    Returns:
        str: commit hash
    """
    label = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    return label.decode("utf-8")


def get_server_name() -> str:
    """Always returns `unknown` for the public code :)

    Returns:
        str: _description_
    """
    return "unknown"


def profile(output_file=None, sort_by="cumulative", lines_to_print=None, strip_dirs=False):
    """A time profiler decorator.
    Inspired by: http://code.activestate.com/recipes/577817-profile-decorator/

    Usage:
    ```
    @profile(output_file= ...)
    def your_function():
        ...
    ```
    Then you will get the profile automatically after the function call is finished.

    Args:
        output_file: str or None. Default is None
            Path of the output file. If only name of the file is given, it's
            saved in the current directory.
            If it's None, the name of the decorated function is used.
        sort_by: str or SortKey enum or tuple/list of str/SortKey enum
            Sorting criteria for the Stats object.
            For a list of valid string and SortKey refer to:
            https://docs.python.org/3/library/profile.html#pstats.Stats.sort_stats
        lines_to_print: int or None
            Number of lines to print. Default (None) is for all the lines.
            This is useful in reducing the size of the printout, especially
            that sorting by 'cumulative', the time consuming operations
            are printed toward the top of the file.
        strip_dirs: bool
            Whether to remove the leading path info from file names.
            This is also useful in reducing the size of the printout
    Returns:
        Profile of the decorated function
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _output_file = output_file or func.__name__ + ".prof"
            pr = cProfile.Profile()
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            pr.dump_stats(_output_file)

            with open(_output_file, "w") as f:
                ps = pstats.Stats(pr, stream=f)
                if strip_dirs:
                    ps.strip_dirs()
                if isinstance(sort_by, (tuple, list)):
                    ps.sort_stats(*sort_by)
                else:
                    ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
            return retval

        return wrapper

    return inner


class SingleThreadInMemoryStorage(optuna.storages.InMemoryStorage):
    """This is faster version of in-memory storage only when the study n_jobs = 1 (single thread).
    Adopted from https://github.com/optuna/optuna/issues/3151

    Args:
        optuna ([type]): [description]
    """

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: optuna.distributions.BaseDistribution,
    ) -> None:
        with self._lock:
            trial = self._get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)

            study_id = self._trial_id_to_study_id_and_number[trial_id][0]
            # Check param distribution compatibility with previous trial(s).
            if param_name in self._studies[study_id].param_distribution:
                optuna.distributions.check_distribution_compatibility(
                    self._studies[study_id].param_distribution[param_name], distribution
                )
            # Set param distribution.
            self._studies[study_id].param_distribution[param_name] = distribution

            # Set param.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution
