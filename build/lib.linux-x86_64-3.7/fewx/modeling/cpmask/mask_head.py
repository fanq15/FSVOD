# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm, NaiveSyncBatchNorm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from fewx.layers import interpolate, get_instances_contour_interior
#from pytorch_toolbelt import losses as L
#from pytorch_toolbelt.modules import AddCoords
import numpy as np

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def mask_rcnn_loss(pred_mask_logits, pred_boundary_logits, w_affinity, instances):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    gt_boundary = []
    #gt_big_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

        boundary_ls = []
        for mask in gt_masks_per_image:
            mask_b = mask.data.cpu().numpy()
            boundary, inside_mask, weight = get_instances_contour_interior(mask_b)
            boundary = torch.from_numpy(boundary).to(device=mask.device).unsqueeze(0)

            boundary_ls.append(boundary)

        #gt_big_masks.append(resized_mask)
        gt_boundary.append(cat(boundary_ls, dim=0))

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0, pred_boundary_logits.sum() * 0, 0
    
    gt_masks = cat(gt_masks, dim=0)

    ####################
    assert len(pred_mask_logits) == len(w_affinity)
    gt_masks_ds = gt_masks.clone().to(dtype=torch.float32)
    gt_masks_ds_inter = F.interpolate(gt_masks_ds.unsqueeze(1), scale_factor=0.5).squeeze().to(dtype=torch.int32).view(gt_masks_ds.size(0),-1)

    loss_aff_list = []
    for i in range(total_num_masks):
        indexs = gt_masks_ds_inter[i].nonzero().squeeze()
        if indexs.nelement() == 0 or len(indexs.size()) == 0:
            print('indexs:', indexs.shape)
            continue

        loss_affinity = w_affinity[i,indexs][:,indexs].sum(1).mean()
        loss_aff_list.append(loss_affinity)

    final_loss_aff = torch.tensor(loss_aff_list)
    gt_aff = torch.ones_like(final_loss_aff)

    final_affinity_loss = nn.L1Loss()(final_loss_aff, gt_aff)
    ##################

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)

    mask_loss = F.binary_cross_entropy_with_logits(
        pred_mask_logits, gt_masks.to(dtype=torch.float32), reduction="mean"
    )

    pred_boundary_logits = pred_boundary_logits[:, 0]
    assert len(pred_mask_logits) == len(pred_boundary_logits)
    gt_boundary = cat(gt_boundary, dim=0)
    boundary_loss = F.binary_cross_entropy_with_logits(
        pred_boundary_logits, gt_boundary.to(dtype=torch.float32), reduction="mean"
    )

    return {'loss_mask': mask_loss, 'loss_boundary': boundary_loss * 0.5, 'loss_affinity': final_affinity_loss * 0.5}

def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


@ROI_MASK_HEAD_REGISTRY.register()
class CPMaskConvUpsampleHead(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(CPMaskConvUpsampleHead, self).__init__()

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        # our settings
        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)


        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)

        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

        # add gcn
        self.query_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, activation=F.relu) #, bias=False, norm=get_norm('SyncBN', input_channels), activation=F.relu)
        self.key_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, activation=F.relu) #, bias=False, norm=get_norm('SyncBN', input_channels), activation=F.relu)
        self.value_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, activation=F.relu) #, bias=False, norm=get_norm('SyncBN', input_channels), activation=F.relu)
        self.output_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, activation=F.relu) #, bias=False, norm=get_norm('SyncBN', input_channels), activation=F.relu)

        self.scale = 1.0 / (input_channels ** 0.5)

        self.blocker = nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized
        #self.blocker = NaiveSyncBatchNorm(input_channels, eps=1e-04) #nn.BatchNorm2d(input_channels, eps=1e-04) # should be zero initialized

        for layer in [self.query_transform, self.key_transform, self.value_transform, self.output_transform]:
            weight_init.c2_msra_fill(layer)

        # add boundary
        self.boundary_conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("boundary_fcn{}".format(k + 1), conv)
            self.boundary_conv_norm_relus.append(conv)

        self.boundary_deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.boundary_predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.boundary_conv_norm_relus + [self.boundary_deconv]:
            weight_init.c2_msra_fill(layer)

        nn.init.normal_(self.boundary_predictor.weight, std=0.001)
        if self.boundary_predictor.bias is not None:
            nn.init.constant_(self.boundary_predictor.bias, 0)

        # add boundary gate
        self.boundary_gate = Conv2d(input_channels*2, 1, kernel_size=1, stride=1, padding=0) #, bias=False, norm=get_norm('SyncBN', 1))
        self.mask_transform = Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False, norm=get_norm('GN', input_channels), activation=F.relu)

        for layer in [self.boundary_gate, self.mask_transform]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        B, C, H, W = x.size()


        # add boundary
        x_boundary = x.clone()

        for layer in self.boundary_conv_norm_relus:
            x_boundary = layer(x_boundary)

        x = x + x_boundary * F.sigmoid(self.boundary_gate(torch.cat((x, x_boundary), dim=1)))
        x = self.mask_transform(x)

        for cnt, layer in enumerate(self.conv_norm_relus):

            x = layer(x)

            if cnt == 3:
                # add gcn
                # x: B,C,H,W
                # x_query: B,C,HW
                x_query = self.query_transform(x).view(B, C, -1)
                x_query_std, x_query_mean = torch.std_mean(x_query, dim=1, keepdim=True)
                x_query = (x_query - x_query_mean) / x_query_std

                # x_query: B,HW,C
                x_query = torch.transpose(x_query, 1, 2)
                # x_key: B,C,HW
                x_key = self.key_transform(x).view(B, C, -1)
                x_key_std, x_key_mean = torch.std_mean(x_key, dim=1, keepdim=True)
                x_key = (x_key - x_key_mean) / x_key_std

                # x_value: B,C,HW
                x_value = self.value_transform(x).view(B, C, -1)
                # x_value: B,HW,C
                x_value = torch.transpose(x_value, 1, 2)
                # W = Q^T K: B,HW,HW
                x_w = torch.matmul(x_query, x_key) * self.scale
                x_w = F.softmax(x_w, dim=-1)
                # x_relation = WV: B,HW,C
                x_relation = torch.matmul(x_w, x_value)
                # x_relation = B,C,HW
                x_relation = torch.transpose(x_relation, 1, 2)
                # x_relation = B,C,H,W
                x_relation = x_relation.view(B,C,H,W)
                x_relation = self.output_transform(x_relation)
                x_relation = self.blocker(x_relation)

                x = x + x_relation

        x_mask = F.relu(self.deconv(x))
        mask = self.predictor(x_mask)

        # add boundary
        x_boundary = F.relu(self.boundary_deconv(x_boundary))
        boundary = self.boundary_predictor(x_boundary)

        # with gcn_affinity
        return mask, boundary, x_w.view(B, -1, H*W)

def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
