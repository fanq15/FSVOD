  
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
import json

def evaluate_predictions_on_coco(
    coco_gt, coco_results, iou_type
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0


    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


gt_file = 'datasets/fsvod/annotations/fsvod_val.json'
pred_file = 'output/final_tpn_results.json'
coco_gt = COCO(gt_file)
iou_type = 'bbox'
with open(pred_file, 'r') as f:
    coco_results = json.load(f)
evaluate_predictions_on_coco(coco_gt, coco_results, iou_type)
