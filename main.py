# type: ignore
import argparse
import logging
import os
import shutil
import sys

import numpy as np
import yaml
from tqdm import tqdm

from src import data_loader, solver, utils, visualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/mvsec_indoor_no_timeaware.yaml",
        help="Config file yaml path",
        type=str,
    )
    parser.add_argument(
        "--eval",
        help="Add for evaluation run",
        action="store_true",
    )
    parser.add_argument(
        "--log", help="Log level: [debug, info, warning, error, critical]", type=str, default="info"
    )
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    return config, args


def save_config(save_dir: str, file_name: str, log_level=logging.INFO):
    """Save configuration"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy(file_name, save_dir)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(f"{save_dir}/main.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def evaluate_mvsec_dataset_with_gt(eval_frame_time_stamp_list, data_config, loader, solv):
    logger.info("Evaluation pipeline")
    eval_dt = data_config["eval_dt"]
    assert eval_dt == 1 or eval_dt == 4
    logger.info(f"dt (for MVSEC) is {eval_dt}")
    n_events = data_config["n_events_per_batch"]

    for i1 in tqdm(range(len(eval_frame_time_stamp_list) - eval_dt)):
        logger.info(f"Frame {i1} of {len(eval_frame_time_stamp_list)}")
        try:
            if i1 < data_config["ind1"] or i1 > data_config["ind2"]:
                continue  # cutofff
        except KeyError:
            pass
        t1 = eval_frame_time_stamp_list[i1]
        t2 = eval_frame_time_stamp_list[i1 + eval_dt]
        ind1 = loader.time_to_index(t1)  # event index
        ind2 = loader.time_to_index(t2)

        # Flow error metrics calculation is based on GT flow + events between the consective GT flow frames
        batch_for_gt_slice = loader.load_event(ind1, ind2)
        gt_flow = loader.load_optical_flow(t1, t2)
        flow_time = t2 - t1
        batch_for_gt_slice[..., 2] -= np.min(batch_for_gt_slice[..., 2])

        # Optimization is based on fixed number of events
        if ind2 - ind1 < n_events:
            logger.info(
                f"Less events in one GT flow sequence. Events: {ind2-ind1} / Expected: {n_events}"
            )
            insufficient = n_events - (ind2 - ind1)
            ind1 -= insufficient // 2
            ind2 += insufficient // 2
        elif ind2 - ind1 > n_events:
            logger.info(
                f"Too many events in one GT flow sequence. Events: {ind2-ind1} / Expected: {n_events}"
            )
            ind1 = ind2 - n_events

        batch_for_optimization = loader.load_event(max(ind1, 0), min(ind2, len(loader)))
        batch_for_optimization[..., 2] -= np.min(batch_for_optimization[..., 2])

        if utils.check_key_and_bool(data_config, "remove_car"):
            logger.info("Remove car-boody pixels")
            batch_for_optimization = utils.crop_event(batch_for_optimization, 0, 193, 0, 346)

        best_motion = solv.optimize(batch_for_optimization)
        solv.set_previous_frame_best_estimation(best_motion)
        # mask with event
        flow_error_with_mask = solv.calculate_flow_error(best_motion, gt_flow, timescale=flow_time, events=batch_for_gt_slice)  # type: ignore
        solv.save_flow_error_as_text(i1, flow_error_with_mask, "flow_error_per_frame_with_mask.txt")  # type: ignore

        # Visualization
        solv.visualize_original_sequential(batch_for_gt_slice)
        solv.visualize_pred_sequential(batch_for_gt_slice, best_motion)
        solv.visualize_gt_sequential(batch_for_gt_slice, gt_flow)


if __name__ == "__main__":
    config, args = parse_args()
    data_config: dict = config["data"]
    out_config: dict = config["output"]
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError("Invalid log level: %s" % log_level)
    save_config(out_config["output_dir"], args.config_file, log_level)
    logger = logging.getLogger(__name__)

    if utils.check_key_and_bool(config, "fix_random_seed"):
        utils.fix_random_seed()

    # Visualizer
    image_shape = (data_config["height"], data_config["width"])
    if config["is_dnn"] and "crop" in data_config["preprocess"].keys():
        image_shape = (data_config["preprocess"]["crop"]["height"], data_config["preprocess"]["crop"]["width"])  # type: ignore

    viz = visualizer.Visualizer(
        image_shape,
        show=out_config["show_interactive_result"],
        save=True,
        save_dir=out_config["output_dir"],
    )

    # Loader
    loader = data_loader.collections[data_config["dataset"]](config=data_config)
    loader.set_sequence(data_config["sequence"])

    # Solver
    method_name = config["solver"]["method"]
    solv: solver.SolverBase = solver.collections[method_name](
        image_shape,
        calibration_parameter=loader.load_calib(),
        solver_config=config["solver"],
        optimizer_config=config["optimizer"],
        output_config=config["output"],
        visualize_module=viz,
    )

    if args.eval:  # Run evaluation piipeline.
        if config["is_dnn"]:
            e = "DNN code is not published."
            logger.error(e)
            raise NotImplementedError(e)
        else:
            logger.info("Sequential optimization")
            assert loader.gt_flow_available  # evaluate with GT flow
            logger.info("evaluation with GT")
            eval_frame_time_stamp_list = loader.eval_frame_time_list()
            evaluate_mvsec_dataset_with_gt(eval_frame_time_stamp_list, data_config, loader, solv)
            logger.info(f"Evaluation done! {data_config['sequence']}")
        exit()

    # Not evaluation - single frame optimization
    if config["is_dnn"]:
        e = "DNN code is not published."
        logger.error(e)
        raise NotImplementedError(e)
    else:  # For non-DNN method
        logger.info("Single-frame optimization")
        ind1, ind2 = data_config["ind1"], data_config["ind2"]
        batch: np.ndarray = loader.load_event(ind1, ind2)
        batch[..., 2] -= np.min(batch[..., 2])

        if utils.check_key_and_bool(data_config, "remove_car"):
            batch = utils.crop_event(batch, 0, 193, 0, 346)  # remvoe MVSEC car

        solv.visualize_one_batch_warp(batch)
        best_motion: np.ndarray = solv.optimize(batch)
        solv.visualize_one_batch_warp(batch, best_motion)

        # Calculate Flow error when GT is available
        if loader.gt_flow_available:
            t1 = loader.index_to_time(ind1)
            t2 = loader.index_to_time(ind2)
            gt_flow = loader.load_optical_flow(t1, t2)

            solv.visualize_one_batch_warp_gt(batch, gt_flow)
            solv.calculate_flow_error(best_motion, gt_flow, t2 - t1, batch)
