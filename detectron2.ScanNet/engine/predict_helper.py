# Copyright (c) Facebook, Inc. and its affiliates.
import atexit
import bisect
import cv2
import torch
import os
import sys

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from engine.customized import ScanNetPredictor


class PredictHelper(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = ScanNetPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        
        features = self.predictor(image)

        return features
