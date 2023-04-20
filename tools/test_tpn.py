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
#from lib.nms import cython_soft_nms_wrapper

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
from detectron2.structures import Boxes, Instances, pairwise_iou
from tao.toolkit.tao import TaoEval
import json
import logging

# constants
WINDOW_NAME = "COCO detections"
color_list = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (128, 0, 255), 
(0, 128, 255), (128, 255, 0), (0, 255, 128), (255, 128, 0), (255, 0, 128), (128, 128, 255), (128, 255, 128), (255, 128, 128), (128, 128, 0), (128, 0, 128)]



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
        #print(self.model)
        #if len(cfg.DATASETS.TEST):
        #    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, pair_image, last_feat, last_box):
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

            inputs = {"image": image, "pair_image": pair_image, "height": height, "width": width, "last_feat": last_feat, "last_box": last_box, "relation_flag": False} #, "annotations": [{"category_id": category_id}]}
            predictions = self.model([inputs]) #[0]
            return predictions

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
    id_0 = instances_0.id_label
    id_1 = instances_1.id_label

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

def generate_results(predictions, gt_labels, pred_txt, gt_txt, img_cnt):
    # only use the first prediction
    cur_predictions = predictions[0]
    instances_0 = cur_predictions['instances'].to(torch.Tensor([1.]))
    boxes_0 = instances_0.pred_boxes.tensor
    scores_0 = instances_0.scores
    id_0 = instances_0.id_label

    new_boxes = boxes_0.clone()
    new_boxes[:, 2] = boxes_0[:, 2] - boxes_0[:, 0]
    new_boxes[:, 3] = boxes_0[:, 3] - boxes_0[:, 1]
    instances_0.pred_boxes = Boxes(new_boxes)

    # generate gt_boxes and gt_category_id
    video_ls = []
    image_ls = []
    track_ls = []
    bbox_ls = []
    category_ls = []
    gt_ls = []
    for item in gt_labels:
        video_id = item['video_id']
        image_id = item['image_id']
        track_id = item['track_id']
        ori_bbox = item['bbox']
        category_id = item['category_id']

        # save gt_dict
        gt_dict = {}
        gt_dict['video_id'] = video_id
        gt_dict['image_id'] = image_id
        gt_dict['track_id'] = track_id
        gt_dict['bbox'] = ori_bbox
        gt_dict['category_id'] = category_id
        gt_dict['score'] = 1.0
        gt_ls.append(gt_dict)

        bbox = ori_bbox.copy()
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]

        video_ls.append(video_id)
        image_ls.append(image_id)
        track_ls.append(track_id)
        bbox_ls.append(bbox)
        category_ls.append(category_id)

        gt_txt += str(img_cnt) + ',' + str(track_id) + ',' + str(ori_bbox[0]) + ',' + str(ori_bbox[1]) + ',' + str(ori_bbox[2]) + ',' + str(ori_bbox[3]) + ',1,1,1\n'


    if len(set(video_ls)) != 1:
        return []

    assert len(set(video_ls)) == 1
    assert len(set(image_ls)) == 1

    id_0 = category_ls
    track_0 = track_ls
    picked_ls = []
    picked_cls = []
    picked_track = []
    if boxes_0.shape[0] > 0 and len(bbox_ls) > 0:
        match_quality_matrix = pairwise_iou(Boxes(torch.Tensor(bbox_ls)), Boxes(boxes_0))
        buff_0 = [0 for i in range(len(bbox_ls))]
        id_dict = {}
        for i in range(match_quality_matrix.shape[1]):
            # i is the index of boxes_1
            cur_matrix = match_quality_matrix[:, i]

            cur_iou = float(torch.max(cur_matrix).data.cpu().numpy())
            # index_0 is the index of boxes_0
            index_0 = int(torch.argmax(cur_matrix).data.cpu().numpy())
            if buff_0[index_0] < cur_iou:
                buff_0[index_0] = cur_iou
                id_dict[index_0] = i
        if len(id_dict) > 0:
            for key, value in id_dict.items():
                cur_iou = buff_0[key]
                if cur_iou > 0.0:
                    picked_ls.append(value)
                    picked_cls.append(id_0[key])
                    picked_track.append(track_0[key])

    new_instances = instances_0[picked_ls]
    new_instances.pred_classes = torch.Tensor(picked_cls)
    new_instances.pred_tracks = torch.Tensor(picked_track)

    image_id = image_ls[0]
    video_id = video_ls[0]
    bbox_ls = new_instances.pred_boxes.tensor.data.numpy()
    category_ls = new_instances.pred_classes.data.numpy()
    scores_ls = new_instances.scores.data.numpy()
    tracks_ls = new_instances.pred_tracks.data.numpy()

    result_ls = []
    for cnt, bbox in enumerate(bbox_ls):
        cur_dict = {}
        cur_dict['image_id'] = image_id
        cur_dict['category_id'] = int(category_ls[cnt])
        cur_dict['bbox'] = [int(i) for i in bbox.tolist()]
        cur_dict['score'] = float(scores_ls[cnt])
        cur_dict['track_id'] = int(tracks_ls[cnt])
        cur_dict['video_id'] = int(video_id)
        result_ls.append(cur_dict)

        pred_txt += str(img_cnt) + ',' + str(int(new_instances.id_label.data.numpy()[cnt])) + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(bbox[2]) + ',' + str(bbox[3]) + ',-1,-1,-1,-1\n'
    #print(result_ls, gt_labels)
    return result_ls, gt_ls, pred_txt, gt_txt

def save_dict(query_feat, query_result, predictions, save_dir, img_id):
    query_feat = query_feat.detach().cpu().data
    pred_boxes = query_result.pred_boxes.tensor.detach().cpu().data
    scores = query_result.scores.detach().cpu().data
    id_label = query_result.id_label.detach().cpu().data

    pred_dict = {}
    pred_dict['query_feat'] = query_feat
    pred_dict['pred_boxes'] = pred_boxes # x1, y1, x2, y2
    pred_dict['scores'] = scores
    pred_dict['id_label'] = id_label
    pred_dict['real_boxes'] = predictions[0]['instances'].to(torch.Tensor([1.])).pred_boxes.tensor

    save_name = os.path.join(save_dir, str(img_id) + '_' + 'query_feat.pkl')
    torch.save(pred_dict, save_name)


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
        img_id = item['id']
        if file_dir not in img_dict.keys():
            img_dict[file_dir] = [file_name + ':' + str(img_id)]
        else:
            img_dict[file_dir].append(file_name + ':' + str(img_id))
           
    predictor = DefaultPredictor(cfg)

    # sort image 
    for key, value in img_dict.items():
        value = sorted(value)
        img_dict[key] = value

    # for save results
    results_ls = []
    gt_ls = []
    video_cnt = 0
    dir_txt = ''
    for key, value in img_dict.items():
        #if 'GOT-10k_Train_002815' not in key:
        #    continue
        #if 'TAO' not in key:
        #    continue
        img_dir = key
        img_paths = value
        img_num = len(img_paths)

        last_feat = None
        last_box = None
        start_time = time.time()
        pred_txt = ''
        gt_txt = ''

        save_dir = os.path.join('./mot_dir', img_dir)
        dir_txt += save_dir + '\n'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(os.path.join(save_dir, 'gt')):
            os.makedirs(os.path.join(save_dir, 'gt'))

        for cnt, img_path in enumerate(img_paths):
            if cnt == 0:
                last_frame = None
            path, img_id = img_path.split(':')

            # load annotations for this image
            ann_ids = tao.get_ann_ids(img_ids=[int(img_id)])
            gt_labels = tao.load_anns(ann_ids)

            if len(gt_labels) == 0:
                continue

            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")

            # pair image
            pair_img_id = cnt + 1
            if pair_img_id >= img_num:
                pair_img_id = cnt
            pair_img = read_image(img_paths[pair_img_id].split(':')[0], format="BGR")

            predictions, last_feat, last_box, query_feat, query_result = predictor(img, pair_img, last_feat, last_box)

            # we only save the first image
            save_dict(query_feat, query_result, predictions, save_dir, img_id)

            # save results
            cur_results, gt_results, pred_txt, gt_txt = generate_results(predictions, gt_labels, pred_txt, gt_txt, cnt+1)
            results_ls += cur_results # cur_results is a list containing some boxes in the image
            gt_ls += gt_results
            # visualization
            #visualize(predictions, path, img_paths[pair_img_id].split(':')[0], cnt)

        txt_name = save_dir + '.txt'
        gt_txt_name = os.path.join(save_dir,  'gt/gt.txt')

        with open(txt_name, 'w') as f:
            f.write(pred_txt)
        with open(gt_txt_name, 'w') as f:
            f.write(gt_txt)

        '''
        save_dir = os.path.join('./gt_mot_dir', img_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(os.path.join(save_dir, 'gt')):
            os.makedirs(os.path.join(save_dir, 'gt'))

        txt_name = save_dir + '.txt'
        gt_txt_name = os.path.join(save_dir,  'gt/gt.txt')
        with open(txt_name, 'w') as f:
            f.write(gt_txt.replace(',1,1,1\n', ',-1,-1,-1,-1\n'))
        with open(gt_txt_name, 'w') as f:
            f.write(gt_txt)
        '''

        video_cnt += 1
        logger.info(
            "{}: {} in {:.2f}s".format(
                str(video_cnt) + ': ' +  img_dir,
                "detected {} images".format(img_num),
                time.time() - start_time,
            )
        )

    with open('./mot_dir/path.txt', 'w') as f:
        f.write(dir_txt)

    save_name = './output/tpn_results.json'
    with open(save_name, 'w') as f:
        json.dump(results_ls, f)

    save_name = './output/tpn_gt.json'
    with open(save_name, 'w') as f:
        json.dump(gt_ls, f)

