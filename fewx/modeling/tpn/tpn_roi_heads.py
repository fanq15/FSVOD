# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
#from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
#from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

# add my library
#from .tpn_proposal_utils import add_ground_truth_to_proposals
from .tpn_fast_rcnn import FastRCNNOutputLayers
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
import fvcore.nn.weight_init as weight_init
from torch.nn import functional as F
from fewx.layers import DFConv2d
from ..pooler import ROIPooler
from mmcv.cnn.bricks import NonLocal2d
from .fsod_fast_rcnn import FsodFastRCNNOutputLayers
#import ipdb

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It typically contains logic to

    1. (in training only) match proposals with ground truth and sample them
    2. crop the regions and extract per-region features using proposals
    3. make per-region predictions with different heads

    It can have many variants, implemented as subclasses of this class.
    This base class contains the logic to match/sample proposals.
    But it is not necessary to inherit this class if the sampling logic is not needed.
    """

    @configurable
    def __init__(
        self,
        *,
        num_classes,
        batch_size_per_image,
        positive_fraction,
        proposal_matcher,
        proposal_append_gt=True
    ):
        """
        NOTE: this interface is experimental.

        Args:
            num_classes (int): number of classes. Used to label background proposals.
            batch_size_per_image (int): number of proposals to sample for training
            positive_fraction (float): fraction of positive (foreground) proposals
                to sample for training.
            proposal_matcher (Matcher): matcher that matches proposals and ground truth
            proposal_append_gt (bool): whether to include ground truth as proposals as well
        """
        super().__init__()
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    @classmethod
    def from_config(cls, cfg):
        return {
            "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "proposal_append_gt": cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT,
            # Matcher to assign box proposals to gt boxes
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),
        }

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.
        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.
        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        #return sampled_idxs, gt_classes[sampled_idxs]
        return sampled_idxs, gt_classes

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.
        Args:
            See :meth:`ROIHeads.forward`
        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)
                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        # proposals and targets belong to the same image
        # in: inside
        # proposals_0 and targets_0
        # proposals_1 and targets_1
        # proposals_boxes and objectness_logits are shared
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes_inside = gt_classes[sampled_idxs] # we sample gt_classes here

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                proposals_per_image.gt_boxes_inside = proposals_per_image.gt_boxes
                proposals_per_image.remove("gt_boxes")
                proposals_per_image.gt_tracks_inside = proposals_per_image.gt_tracks
                proposals_per_image.remove("gt_tracks")

                proposals_per_image.remove("gt_classes")
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes_inside = gt_boxes
                proposals_per_image.gt_tracks_inside = gt_classes[sampled_idxs]


            proposals_with_gt.append(proposals_per_image)

        # proposals and targets belong to different images
        # out: outside
        # proposals_0 and targets_1
        # proposals_1 and targets_0
        final_proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals_with_gt, targets[::-1]):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )

            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image #[sampled_idxs]
            proposals_per_image.gt_classes_outside = gt_classes #[sampled_idxs] we do not sample here

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs #[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
                proposals_per_image.gt_boxes_outside = proposals_per_image.gt_boxes
                proposals_per_image.remove("gt_boxes")

                proposals_per_image.gt_tracks_outside = proposals_per_image.gt_tracks
                proposals_per_image.remove("gt_tracks")

                proposals_per_image.remove("gt_classes")
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(gt_classes), 4))
                )
                proposals_per_image.gt_boxes_outside = gt_boxes
                proposals_per_image.gt_tracks_outside = gt_classes

            # sames
            track_same = proposals_per_image.gt_tracks_outside == proposals_per_image.gt_tracks_inside
            object_0 = 0 == proposals_per_image.gt_classes_outside
            object_1 = 0 == proposals_per_image.gt_classes_inside
            objectness_same = object_0 * object_1
            sames = track_same * objectness_same
            sames = ~sames

            proposals_per_image.gt_sames = sames.to(gt_classes)

            final_proposals_with_gt.append(proposals_per_image)

        # rearrange labels
        final_proposals_with_gt[0].gt_classes_0 = final_proposals_with_gt[0].gt_classes_inside
        final_proposals_with_gt[0].gt_boxes_0 = final_proposals_with_gt[0].gt_boxes_inside
        final_proposals_with_gt[0].gt_tracks_0 = final_proposals_with_gt[0].gt_tracks_inside

        final_proposals_with_gt[0].gt_classes_1 = final_proposals_with_gt[0].gt_classes_outside
        final_proposals_with_gt[0].gt_boxes_1 = final_proposals_with_gt[0].gt_boxes_outside
        final_proposals_with_gt[0].gt_tracks_1 = final_proposals_with_gt[0].gt_tracks_outside

        final_proposals_with_gt[1].gt_classes_0 = final_proposals_with_gt[1].gt_classes_outside
        final_proposals_with_gt[1].gt_boxes_0 = final_proposals_with_gt[1].gt_boxes_outside
        final_proposals_with_gt[1].gt_tracks_0 = final_proposals_with_gt[1].gt_tracks_outside

        final_proposals_with_gt[1].gt_classes_1 = final_proposals_with_gt[1].gt_classes_inside
        final_proposals_with_gt[1].gt_boxes_1 = final_proposals_with_gt[1].gt_boxes_inside
        final_proposals_with_gt[1].gt_tracks_1 = final_proposals_with_gt[1].gt_tracks_inside


        final_proposals_with_gt = Instances.cat(final_proposals_with_gt)
        final_proposals_with_gt.remove('gt_classes_inside')
        final_proposals_with_gt.remove('gt_boxes_inside')
        final_proposals_with_gt.remove('gt_tracks_inside')
        final_proposals_with_gt.remove('gt_classes_outside')
        final_proposals_with_gt.remove('gt_boxes_outside')
        final_proposals_with_gt.remove('gt_tracks_outside')

        #final_proposals_with_gt = [final_proposals_with_gt]
        #final_proposals_with_gt = [final_proposals_with_gt, final_proposals_with_gt]
        final_proposals_with_gt_1 = Instances(final_proposals_with_gt.image_size)
        final_proposals_with_gt_1.gt_classes_0 = final_proposals_with_gt.gt_classes_0
        final_proposals_with_gt_1.gt_boxes_0 = final_proposals_with_gt.gt_boxes_0
        final_proposals_with_gt_1.gt_tracks_0 = final_proposals_with_gt.gt_tracks_0
        final_proposals_with_gt_1.gt_classes_1 = final_proposals_with_gt.gt_classes_1
        final_proposals_with_gt_1.gt_boxes_1 = final_proposals_with_gt.gt_boxes_1
        final_proposals_with_gt_1.gt_tracks_1 = final_proposals_with_gt.gt_tracks_1
        final_proposals_with_gt_1.proposal_boxes = final_proposals_with_gt.proposal_boxes
        final_proposals_with_gt_1.objectness_logits = final_proposals_with_gt.objectness_logits
        final_proposals_with_gt_1.gt_sames = final_proposals_with_gt.gt_sames

        final_proposals_with_gt.gt_classes = final_proposals_with_gt.gt_classes_0
        final_proposals_with_gt.gt_boxes = final_proposals_with_gt.gt_boxes_0

        final_proposals_with_gt_1.gt_classes = final_proposals_with_gt_1.gt_classes_1
        final_proposals_with_gt_1.gt_boxes = final_proposals_with_gt_1.gt_boxes_1

        full_final_proposals_with_gt = [final_proposals_with_gt, final_proposals_with_gt_1]

        # resample for relation net
        relation_proposals_0 = Instances(final_proposals_with_gt.image_size)
        relation_proposals_0.gt_classes = final_proposals_with_gt.gt_classes_0
        relation_proposals_0.gt_boxes = final_proposals_with_gt.gt_boxes_0
        relation_proposals_0.proposal_boxes = final_proposals_with_gt.proposal_boxes
        relation_proposals_0.objectness_logits = final_proposals_with_gt.objectness_logits
        
        relation_proposals_1 = Instances(final_proposals_with_gt_1.image_size)
        relation_proposals_1.gt_classes = final_proposals_with_gt_1.gt_classes_1
        relation_proposals_1.gt_boxes = final_proposals_with_gt_1.gt_boxes_1
        relation_proposals_1.proposal_boxes = final_proposals_with_gt_1.proposal_boxes
        relation_proposals_1.objectness_logits = final_proposals_with_gt_1.objectness_logits

        relation_proposals = [relation_proposals_0, relation_proposals_1]

        relation_proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(relation_proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

                proposals_per_image.remove("gt_tracks")
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes


            relation_proposals_with_gt.append(proposals_per_image)

        return full_final_proposals_with_gt, relation_proposals_with_gt
 
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class TpnStandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_pooler_fc: ROIPooler,
        offset_fc: nn.Module,
        #refine: nn.Module,
        relation_pooler: ROIPooler,
        relation_predictor: nn.Module,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.train_on_pred_boxes = train_on_pred_boxes
        #self.share_head = ShareFastRCNNConvHead()
        self.box_pooler_fc = box_pooler_fc
        self.offset_fc = offset_fc
        #self.refine = refine
        self.relation_pooler = relation_pooler
        self.relation_predictor = relation_predictor
        self.criterion = nn.CrossEntropyLoss()
        self.support_fc_1 = nn.Linear(2048*7*7, 1024)
        self.support_fc_2 = nn.Linear(1024, 320)
        nn.init.normal_(self.support_fc_1.weight, std=0.01)
        nn.init.constant_(self.support_fc_1.bias, 0)
        nn.init.normal_(self.support_fc_2.weight, std=0.01)
        nn.init.constant_(self.support_fc_2.bias, 0)
        self.avgpool = nn.AvgPool2d(7)

        self.conv_trans = nn.Conv2d(2048, 512, 1, padding=0)
        nn.init.normal_(self.conv_trans.weight, std=0.01)
        nn.init.constant_(self.conv_trans.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        
        box_pooler_fc = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        offset_fc = nn.Sequential(
            nn.Linear(7 * 7 * 512 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * 7 * 2))
        
        #refine = NonLocal2d(
        #    256,
        #    reduction=1,
        #    use_scale=False)

        relation_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=[pooler_scales[0]],
            sampling_ratio=sampling_ratio,
            pooler_type="ROIAlignV2",
        )

        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=int(in_channels * 2 / 4), height=pooler_resolution, width=pooler_resolution)
        )

        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)

        relation_predictor = FsodFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,

            "box_pooler_fc": box_pooler_fc,
            "offset_fc": offset_fc,
            #"refine": refine,
            "relation_pooler": relation_pooler,
            "relation_predictor": relation_predictor
        }
    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        support_features: Dict[str, torch.Tensor],
        crop_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        support_boxes: List[Boxes],
        crop_boxes: List[Boxes],
        targets: Optional[List[Instances]] = None,
        support_cls: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """

        del images
        if self.training:
            assert targets
            for item in targets:
                assert len(torch.unique(item.gt_classes)) <= 1

            proposals, relation_proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
 
        if self.training:
            losses = self._forward_box(features, support_features, crop_features, proposals, relation_proposals, support_boxes, crop_boxes, support_cls)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            return proposals, losses
        else:
            pred_instances, query_feat = self._forward_box(features, support_features, None, proposals, None, support_boxes, None)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, query_feat #{}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0][0].has("pred_boxes") and instances[0][0].has("pred_classes") #\
               #and instances[0][0].has("sames")

        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], support_features: Dict[str, torch.Tensor], crop_features: Dict[str, torch.Tensor], proposals: List[Instances], relation_proposals: List[Instances], support_boxes: List[Boxes], crop_boxes: List[Boxes], support_cls: Optional[List[Instances]] = None) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]

        new_features = []
        for feat in features:
            feat = F.relu(self.conv_trans(feat), True)
            new_feat_0 = torch.cat([feat[0].unsqueeze(0), feat[1].unsqueeze(0)], dim=1)
            new_feat_1 = torch.cat([feat[1].unsqueeze(0), feat[0].unsqueeze(0)], dim=1)

            new_feat = torch.cat([new_feat_0, new_feat_1], dim=0)
            new_features.append(new_feat)

        box_features_fc = self.box_pooler_fc(new_features, [x.proposal_boxes for x in proposals])
        rois_num = box_features_fc.size(0)
        offset = self.offset_fc(box_features_fc.view(rois_num, -1))

        box_features = self.box_pooler(new_features, [x.proposal_boxes for x in proposals], offset)
        box_features = self.box_head(box_features)

        predictions = self.box_predictor(box_features)
        del box_features

        # few-shot multi-relation head

        # query feature
        # step 1: gather multi-level features by resize and average
        # use p4, index = 1
        feats = []
        num_levels = len(features)
        refine_level = 0 #2
        ori_query_feat = features[refine_level]
        '''
        gather_size = features[refine_level].size()[2:]
        for i in range(num_levels):
            if i < refine_level:
                gathered = F.adaptive_max_pool2d(
                    features[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    features[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        ori_query_feat = sum(feats) / len(feats)
        '''
        # step 2: refine gathered features
        # query feature
        #query_feat = self.refine(bsf)

        if self.training:

            # crop features 
            crop_features = [crop_features[f] for f in self.box_in_features]
            # crop feature
            # step 1: gather multi-level features by resize and average
            # use p4, index = 1
            crop_feats = []
            num_levels = len(crop_features)
            refine_level = 0 #2
            crop_feat = crop_features[refine_level]
            '''
            igather_size = crop_features[refine_level].size()[2:]
            for i in range(num_levels):
                if i < refine_level:
                    crop_gathered = F.adaptive_max_pool2d(
                        crop_features[i], output_size=gather_size)
                else:
                    crop_gathered = F.interpolate(
                        crop_features[i], size=gather_size, mode='nearest')
                crop_feats.append(crop_gathered)

            crop_feat = sum(crop_feats) / len(crop_feats)
            '''
            crop_feat = self.relation_pooler([crop_feat], crop_boxes)


            support_features = [support_features[f] for f in self.box_in_features]
            # support feature
            # step 1: gather multi-level features by resize and average
            # use p4, index = 1
            support_feats = []
            num_levels = len(support_features)
            refine_level = 0 #2
            support_feat = support_features[refine_level]
            '''
            gather_size = support_features[refine_level].size()[2:]
            for i in range(num_levels):
                if i < refine_level:
                    support_gathered = F.adaptive_max_pool2d(
                        support_features[i], output_size=gather_size)
                else:
                    support_gathered = F.interpolate(
                        support_features[i], size=gather_size, mode='nearest')
                support_feats.append(support_gathered)

            support_feat = sum(support_feats) / len(support_feats)
            '''
            # step 2: refine gathered features
            # support feature
            #support_feat = self.refine(support_bsf)
            relation_loss_cls = []
            relation_loss_box_reg = []

            support_feat = self.relation_pooler([support_feat], support_boxes)
            x = torch.flatten(support_feat, 1)
            x = F.relu(self.support_fc_1(x), True)
            support_pred = self.support_fc_2(x)
            support_cls_loss = self.criterion(support_pred, support_cls)
            support_loss = {'support_cls_loss': support_cls_loss}

            for i in range(len(relation_proposals)):
                #ipdb.set_trace()
                # few-shot relation branch

                pos_proposals = relation_proposals[i]
                query_feat = self.relation_pooler([ori_query_feat[i].unsqueeze(0)], [pos_proposals.proposal_boxes])

                # add crop features to each fg proposals
                for cnt_i in range(sum(pos_proposals.gt_classes==0)):
                    if np.random.uniform() > 0.2: 
                        pos_idxs = pos_proposals.gt_classes == 0
                        #neg_idxs = pos_proposals.gt_classes == 1

                        crop_num = crop_feat.shape[0]
                        crop_top = int(crop_num / 3.0)
                        if crop_top > 0:
                            crop_shot = np.random.randint(1, crop_top + 1)
                        else:
                            crop_shot = 1
                        crop_idxs = np.random.choice(crop_num, crop_shot, replace=False)
                        query_feat[pos_idxs][cnt_i] = query_feat[pos_idxs][cnt_i] * 0.5 + crop_feat[crop_idxs].mean(0, True) * 0.5
                        '''
                        crop_feat_curr = crop_feat[crop_idxs]
                        query_feat_curr = query_feat[pos_idxs][cnt_i].unsqueeze(0)

                        neg_i = np.random.randint(sum(neg_idxs).cpu())
                        neg_query_feat_curr = query_feat[neg_idxs][neg_i].unsqueeze(0)
                        cat_feat = torch.cat([query_feat_curr, neg_query_feat_curr, crop_feat_curr], dim=0)

                        
                        cat_feat_avg = self.avgpool(cat_feat).squeeze(-1).squeeze(-1)

                        #ipdb.set_trace()
                        proto_feat = cat_feat_avg.mean(0, True).expand_as(cat_feat_avg)

                        sim = F.cosine_similarity(cat_feat_avg, proto_feat) * 3.0
                        pos_attention = sim.softmax(0)

                        pos_attention = pos_attention.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(cat_feat)
                        cat_feat = cat_feat * pos_attention
                        query_feat[pos_idxs][cnt_i] = cat_feat.sum(0, True)
                        '''        
                #query_feat = self.share_head(query_feat)
                #support_feat = self.share_head(support_feat)

                # positive support branch ##################################
                pos_begin = i * 5 #i * self.support_shot * self.support_way
                pos_end = pos_begin + 5 #self.support_shot
                pos_support_feat = support_feat[pos_begin:pos_end].mean(0, True)
                pos_pred_class_logits, pos_pred_proposal_deltas = self.relation_predictor(query_feat, pos_support_feat)
                #ipdb.set_trace()
                # negative support branch ##################################
                neg_begin = pos_end + 5
                neg_end = neg_begin + 5 #self.support_shot 
                neg_support_feat = support_feat[neg_begin:neg_end].mean(0, True)

                neg_pred_class_logits, neg_pred_proposal_deltas = self.relation_predictor(query_feat, neg_support_feat)

                #pos_proposals = Instances.cat(proposals)
                neg_proposals = Instances(pos_proposals.image_size)
                neg_proposals.proposal_boxes = pos_proposals.proposal_boxes
                neg_proposals.objectness_logits = pos_proposals.objectness_logits
                neg_proposals.gt_classes = torch.full_like(pos_proposals.gt_classes, 1)
                neg_proposals.gt_boxes = pos_proposals.gt_boxes
                '''
                neg_proposals.gt_sames = pos_proposals.gt_sames
                neg_proposals.gt_classes_0 = pos_proposals.gt_classes_0
                neg_proposals.gt_boxes_0 = pos_proposals.gt_boxes_0
                neg_proposals.gt_tracks_0 = pos_proposals.gt_tracks_0
                neg_proposals.gt_classes_1 = pos_proposals.gt_classes_1
                neg_proposals.gt_boxes_1 = pos_proposals.gt_boxes_1
                neg_proposals.gt_tracks_1 = pos_proposals.gt_tracks_1
                '''
                relation_proposals_curr = [Instances.cat((pos_proposals, neg_proposals))]
                #print(relation_proposals_curr[0].gt_classes)
                # detector loss
                detector_pred_class_logits = torch.cat([pos_pred_class_logits, neg_pred_class_logits], dim=0)
                detector_pred_proposal_deltas = torch.cat([pos_pred_proposal_deltas, neg_pred_proposal_deltas], dim=0)

                #relation_proposals_curr = [Instances.cat(proposals + neg_detector_proposals)]
                relation_predictions = detector_pred_class_logits, detector_pred_proposal_deltas

                # loss
                relation_losses = self.relation_predictor.losses(relation_predictions, relation_proposals_curr)
                relation_loss_cls.append(relation_losses['relation_loss_cls'])
                relation_loss_box_reg.append(relation_losses['relation_loss_box_reg'])

            all_relation_losses = {}
            all_relation_losses['relation_loss_cls'] = torch.stack(relation_loss_cls).mean() 
            all_relation_losses['relation_loss_box_reg'] = torch.stack(relation_loss_box_reg).mean()

            losses = self.box_predictor.losses(predictions, proposals) # before we copy the proposals, now we only use the original proposals
            losses.update(all_relation_losses)
            losses.update(support_loss)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            predictions = predictions[:-1]
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)

            # save query feat
            # we only save the feature of the first image
            ori_query_feat = self.relation_pooler([ori_query_feat[0].unsqueeze(0)], [pred_instances[0][0].pred_boxes])
            #query_feat = self.share_head(query_feat)

            return pred_instances, ori_query_feat

