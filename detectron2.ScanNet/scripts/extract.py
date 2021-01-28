# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
import h5py

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from engine.predict_helper import PredictHelper

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input_dir",
        help="Root directory of ScanNet frames",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="Root directory for the extracted features",
        required=True
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    extractor = PredictHelper(cfg)

    database_path = os.path.join(args.output_dir, "faster_rcnn_R_101_DC5_3x_ScanNet_feats.hdf5")
    database = h5py.File(database_path, "w", libver="latest")

    scan_list = sorted(os.listdir(args.input_dir))
    for scan_id in scan_list:
        image_root = os.path.join(args.input_dir, scan_id, "color")
        image_list = sorted(os.listdir(image_root))
        for image_name in image_list:
            image_id = image_name.split(".")[0]
            image_path = os.path.join(image_root, image_name)
            img = read_image(image_path, format="BGR")
            start_time = time.time()
            features = extractor.run_on_image(img)

            image_key = "{}-{}".format(scan_id, image_id)
            if features != None:
                database.create_dataset(image_key, data=features.cpu().numpy())

            logger.info(
                "{}: {} in {:.2f}s".format(
                    image_key,
                    "extracted features for {} instances".format(features.shape[0])
                    if features != None
                    else "no instance detected",
                    time.time() - start_time,
                )
            )
