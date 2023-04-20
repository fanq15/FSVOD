# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

#TODO : this is a temporary expedient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from detectron2.config import get_cfg
from fewx.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from tao.toolkit.tao import Tao
import torch
import os

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
import numpy as np

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument("--json-path", help="Path to json file.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        print(self.model)
        #if len(cfg.DATASETS.TEST):
        #    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, pair_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
                pair_image = pair_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            pair_image = self.aug.get_transform(pair_image).apply_image(pair_image)
            pair_image = torch.as_tensor(pair_image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "pair_image": pair_image, "height": height, "width": width} #, "annotations": [{"category_id": category_id}]}
            predictions = self.model([inputs]) #[0]
            return predictions

color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), 
(0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]

def draw_caption(image, box, caption, color):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 8), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


def visualize(predictions, img_path_0, img_path_1, img_id):
    predictions_0, predictions_1 = predictions
    instances_0 = predictions_0['instances']
    instances_1 = predictions_1['instances']

    boxes_0 = instances_0.pred_boxes
    boxes_1 = instances_1.pred_boxes
    scores_0 = instances_0.scores
    scores_1 = instances_1.scores
    id_0 = instances_0.id
    id_1 = instances_1.id
    sames_0 = instances_0.sames
    sames_1 = instances_1.sames

    img_dir = os.path.join(*img_path_0.split('/')[3:-1])
    save_dir = os.path.join('./vis', img_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_0 = cv2.imread(img_path_0)
    img_1 = cv2.imread(img_path_1)

    save_path_0 = os.path.join(save_dir, str(img_id) + '_0.jpg')
    save_path_1 = os.path.join(save_dir, str(img_id) + '_1.jpg')

    for box_id, box in enumerate(boxes_0):
        trace_id = id_0[box_id].data.cpu().numpy()
        cur_score = scores_0[box_id].data.cpu().numpy()
        x1, y1, x2, y2 = box
        draw_trace_id = str(trace_id) + '_' + "%.03f" % cur_score
        
        draw_caption(img_0, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
        cv2.rectangle(img_0, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)

    cv2.imwrite(save_path_0, img_0)

    for box_id, box in enumerate(boxes_1):
        trace_id = id_1[box_id].data.cpu().numpy()
        cur_score = scores_1[box_id].data.cpu().numpy()
        x1, y1, x2, y2 = box
        draw_trace_id = str(trace_id) + '_' + "%.03f" % cur_score
        
        draw_caption(img_1, (x1, y1, x2, y2), draw_trace_id, color=color_list[trace_id % len(color_list)])
        cv2.rectangle(img_1, (x1, y1), (x2, y2), color=color_list[trace_id % len(color_list)], thickness=2)

    cv2.imwrite(save_path_1, img_1)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    root_path = 'datasets/fsvod/annotations'
    tao = Tao(os.path.join(root_path, 'fsvod_val.json'))
    img_dict = {}
    for item in tao.load_imgs(None):
        file_name = os.path.join('datasets/fsvod/images', item['file_name'])
        file_dir = os.path.join(*file_name.split('/')[:-1])
        if file_dir not in img_dict.keys():
            img_dict[file_dir] = [file_name]
        else:
            img_dict[file_dir].append(file_name)
           
    predictor = DefaultPredictor(cfg)

    # sort image 
    for key, value in img_dict.items():
        value = sorted(value)
        img_dict[key] = value

    for key, value in img_dict.items():
        img_dir = key
        img_paths = value
        img_num = len(img_paths)

        last_feat = None
        for cnt, path in enumerate(img_paths):
            if cnt == 0:
                last_frame = None
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")

            # pair image
            pair_img_id = cnt + 1
            if pair_img_id >= img_num:
                pair_img_id = cnt
            pair_img = read_image(img_paths[pair_img_id], format="BGR")

            start_time = time.time()
            predictions = predictor(img, pair_img)
            visualize(predictions, path, img_paths[pair_img_id], cnt)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
