# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import math
from typing import List, Tuple
import torch

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances

logger = logging.getLogger(__name__)

def add_ground_truth_to_proposals(gt_boxes, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.
    Args:
        gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (list[Instances]): list of N elements. Element i is a Instances
            representing the proposals for image i.
    Returns:
        list[Instances]: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert gt_boxes is not None

    assert len(proposals) == len(gt_boxes)
    if len(proposals) == 0:
        return proposals

    return [
        add_ground_truth_to_proposals_single_image(gt_boxes_i, proposals_i)
        for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
    ]


def add_ground_truth_to_proposals_single_image(gt_boxes, proposals):
    """
    Augment `proposals` with ground-truth boxes from `gt_boxes`.
    Args:
        Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
        per image.
    Returns:
        Same as `add_ground_truth_to_proposals`, but for only one image.
    """
    
    proposals.proposal_boxes = proposals.pred_boxes
    proposals.remove("pred_boxes")

    proposals.objectness_logits = proposals.scores
    proposals.remove("scores")

    proposals.remove("pred_classes")
    proposals.remove("locations")
    proposals.remove("fpn_levels")
    
    device = proposals.objectness_logits.device
    # Assign all ground-truth boxes an objectness logit corresponding to
    # P(object) = sigmoid(logit) =~ 1.
    gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
    gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

    # Concatenating gt_boxes with proposals requires them to have the same fields
    gt_proposal = Instances(proposals.image_size)
    gt_proposal.proposal_boxes = gt_boxes
    gt_proposal.objectness_logits = gt_logits
    new_proposals = Instances.cat([proposals, gt_proposal])
    del proposals
    return new_proposals
