# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .tpn_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY


from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.matcher import Matcher
from detectron2.layers import batched_nms
import sys
import os
import pandas as pd
import detectron2.data.detection_utils as utils
import torch.nn.functional as F

__all__ = ["TpnRCNN"]


@META_ARCH_REGISTRY.register()
class TpnRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            relation_flag = batched_inputs[0]['relation_flag']

            if relation_flag:
                return self.relation_model(batched_inputs)
            else:
                return self.inference(batched_inputs)

        images, support_images, crop_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                assert len(torch.unique(x['instances'].gt_classes)) <= 1
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            for x in batched_inputs:
                assert len(torch.unique(x['pair_instances'].gt_classes)) <= 1
                x['pair_instances'].set('gt_classes', torch.full_like(x['pair_instances'].get('gt_classes'), 0))
            pair_gt_instances = [x["pair_instances"].to(self.device) for x in batched_inputs]
            gt_instances = gt_instances + pair_gt_instances

            support_cls = torch.Tensor(batched_inputs[0]['support_cls']).to(gt_instances[0].gt_classes)
        else:
            gt_instances = None
            support_cls = None

        features = self.backbone(images.tensor)

        # support branches
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))
        
        B, N, C, H, W = support_images.tensor.shape
        assert N == 2 * 5 * 2 #self.support_way * self.support_shot

        support_images = support_images.tensor.reshape(B*N, C, H, W)
        support_features = self.backbone(support_images)
        
        ##############################################################

        ###################### query crop branch ######################
        # cur image and pair image share the crop_images
        crop_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['crop_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                crop_bboxes_ls.append(box.to(self.device))
        
        B, N, C, H, W = crop_images.tensor.shape
        crop_images = crop_images.tensor.reshape(B*N, C, H, W)
        crop_features = self.backbone(crop_images)
        ###############################################################

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, features, support_features, crop_features, proposals, support_bboxes_ls, crop_bboxes_ls, gt_instances, support_cls)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def relation_model(self, batched_inputs):
        self.support_on = True #False
        support_way = 1
        support_shot = 5 #10

        if not os.path.exists('./support_dir'):
            os.makedirs('./support_dir')

        offline_support_file_name = './support_dir/offline_fsvod_val_support_feature.pkl'
        offline_support_path = './datasets/fsvod/offline_fsvod_val_support_df.pkl'
        #offline_support_file_name = './support_dir/offline_fsvod_test_support_feature.pkl'
        #offline_support_path = './datasets/fsvod/offline_fsvod_test_support_df.pkl'

        if not os.path.exists(offline_support_file_name):

            support_dict = {'support_feat': {}}
            self.offline_df = pd.read_pickle(offline_support_path)
            for cls in self.offline_df['category_id'].unique():
                support_cls_df = self.offline_df.loc[self.offline_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join('./datasets/fsvod', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                support_features = self.backbone(support_images.tensor)

                ####################################################
                # generate support features
                support_features = [support_features[f] for f in self.roi_heads.box_in_features]
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
                #support_feat = self.roi_heads.refine(support_bsf)

                support_feat = self.roi_heads.relation_pooler([support_feat], support_box_all)
                #support_feat = self.roi_heads.share_head(support_feat)

                support_feat = support_feat.mean(0, True)

                support_dict['support_feat'][cls] = support_feat.detach().cpu().data
                del support_feat

            with open(offline_support_file_name, 'wb') as f:
                torch.save(support_dict, f)
            print("=========== Offline support features are generated. ===========")
            print("============ Few-shot object detetion will start. =============")
            sys.exit(0)
        else:

            #print("============ Few-shot object detetion will start. =============")
            self.offline_support_dict  = torch.load(offline_support_file_name)
            for res_key, res_dict in self.offline_support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.offline_support_dict[res_key][cls_key] = feature.cuda()
            
            online_support_path = './datasets/fsvod/online_fsvod_val_support_df.pkl'
            offline_support_path = './datasets/fsvod/offline_fsvod_val_support_df.pkl'

            #online_support_path = './datasets/fsvod/online_fsvod_test_support_df.pkl'
            #offline_support_path = './datasets/fsvod/offline_fsvod_test_support_df.pkl'

            self.online_df = pd.read_pickle(online_support_path)
            self.offline_df = pd.read_pickle(offline_support_path)

            # get query_cls
            online_cls = batched_inputs[0]['online_cls']
            query_video_id = batched_inputs[0]['video_id']
            query_feat = batched_inputs[0]['query_feat'].cuda()

            # get other cls
            support_way = 20 #5
            support_shot = 5

            #online_cls = [online_cls[0]]
            if support_way > 1 and len(set(online_cls)) < support_way:
                all_ls = list(self.offline_support_dict['support_feat'].keys())
                other_cls = list(set(all_ls) - set(online_cls))
                cls_len = len(online_cls)
                pick_num = support_way - cls_len
                picked_cls = np.random.choice(other_cls, pick_num, replace=False).tolist() 

                offline_cls = picked_cls
            else:
                offline_cls = []

            # query_video in offline df?
            # if query_video not in self.offline_df, merge online_cls to offline_cls, set online_cls as empty
            if len(online_cls) != 0:
                if query_video_id not in self.offline_df['video_id']:
                    offline_cls = online_cls + offline_cls
                    online_cls = []
            
            B, _, _, _ = query_feat.shape
            assert B == 1 # only support 1 query image in test

            cls_dict = {}
            # query online mode
            for cls_id in online_cls:
                support_cls_df = self.online_df.loc[(self.online_df['category_id'] == cls_id) & (self.online_df['video_id'] != query_video_id), :].sample(support_shot, random_state=666).reset_index()

                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path = os.path.join('./datasets/fsvod', support_img_df['file_path'])
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                support_features = self.backbone(support_images.tensor)

                ####################################################
                # generate support features
                support_features = [support_features[f] for f in self.roi_heads.box_in_features]
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
                #support_feat = self.roi_heads.refine(support_bsf)

                support_feat = self.roi_heads.relation_pooler([support_feat], support_box_all)
                #support_feat = self.roi_heads.share_head(support_feat)

                support_feat = support_feat.mean(0, True)

                ########################
                # relation network 
                pred_class_logits, pred_proposal_deltas = self.roi_heads.relation_predictor(query_feat, support_feat)

                cls_dict[cls_id] = pred_class_logits.softmax(dim=-1)[:, 0]

            # offline mode 
            for cls_id in offline_cls:
                # support branch ##################################
                support_feat = self.offline_support_dict['support_feat'][cls_id]

                ########################
                # relation network
                pred_class_logits, pred_proposal_deltas = self.roi_heads.relation_predictor(query_feat, support_feat)

                cls_dict[cls_id] = pred_class_logits.softmax(dim=-1)[:, 0]
            #print(cls_dict)
            #pred_cls = max(cls_dict, key=cls_dict.get)
            #cls_score = cls_dict[pred_cls].cpu().data.numpy()[0]

            #return pred_cls, cls_score
            return cls_dict

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images, all_images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        ori_last_feat = batched_inputs[0]['last_feat']
        last_feat = {}
        for key, value in features.items():
            if ori_last_feat is not None:
                features[key] = torch.cat([ori_last_feat[key], value], dim=0)
                last_feat[key] = value
            # first frame
            else:
                last_feat[key] = value[-1, :, :, :].unsqueeze(0)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(all_images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # rename proposal_boxes
            '''
            for item in proposals:
                item.proposal_boxes = item.pred_boxes
                item.remove("pred_boxes")
                item.remove("locations")
                item.remove("fpn_levels")
            '''
            last_box = batched_inputs[0]["last_box"]

            if last_box is not None and len(last_box[0][-1]) > 0:
                last_box = last_box[0][-1]
                id_label = last_box.id_label
                id_max = id_label.max() + 1

                id_label_0 = [i for i in range(id_max, id_max + len(proposals[0]))]
                id_label_1 = [i for i in range(id_max + len(proposals[0]), id_max + len(proposals[0]) + len(proposals[1]))]

                last_box.scores = torch.full_like(last_box.scores, 1.)

                last_box.proposal_boxes = last_box.pred_boxes
                last_box.objectness_logits = last_box.scores
                last_box.remove("pred_boxes")
                last_box.remove("pred_classes")
                last_box.remove("scores")
                #last_box.remove("sames")


                match_quality_matrix = pairwise_iou(last_box.proposal_boxes, proposals[0].proposal_boxes)

                matched_vals, matches = match_quality_matrix.max(dim=0)
                for cnt, cur_iou in enumerate(matched_vals):
                    if cur_iou > 0.3: #0.5:
                        id_label_0[cnt] = id_label[int(matches[cnt].data.cpu().numpy())]

                id_label_0 = torch.Tensor(id_label_0).to(proposals[0].objectness_logits.long())
                proposals[0].id_label = id_label_0
                proposals[0] = Instances.cat((proposals[0], last_box))
                id_label_0 = proposals[0].id_label
                #print(last_box)
            else:
                id_label_0 = [i for i in range(len(proposals[0]))]
                id_label_1 = [i for i in range(len(proposals[0]), len(proposals[0]) + len(proposals[1]))]

                id_label_0 = torch.Tensor(id_label_0).to(proposals[0].objectness_logits.long())
                proposals[0].id_label = id_label_0

            ##################### proposal group #####################
            match_quality_matrix = pairwise_iou(proposals[0].proposal_boxes, proposals[1].proposal_boxes)

            # match_quality_matrix is M (gt) x N (predicted)
            # Max over gt elements (dim 0) to find best gt candidate for each prediction
            matched_vals, matches = match_quality_matrix.max(dim=0)

            for cnt, cur_iou in enumerate(matched_vals):
                if cur_iou > 0.5:
                    id_label_1[cnt] = id_label_0[int(matches[cnt].data.cpu().numpy())]
            
            id_label_1 = torch.Tensor(id_label_1).to(proposals[1].objectness_logits.long())
            proposals[1].id_label = id_label_1
            #if last_box is not None and len(last_box[0][-1]) > 0:
            #    print(last_box.id_label, proposals[0].id_label, proposals[1].id_label)
            # cat proposals
            proposals = Instances.cat(proposals)

            '''
            ##################### NMS between two proposals ##################### 
            # Filter results based on detection scores
            filter_mask = proposals.scores > 0.05  # R x K
            # R' x 2. First column contains indices of the R predictions;
            # Second column contains indices of classes.
            filter_inds = filter_mask.nonzero().squeeze(0)
            keep = batched_nms(proposals.proposal_boxes.tensor, proposals.scores, filter_inds, 0.5)
            proposals = proposals[keep]
            '''
            proposals = [proposals, proposals]
            #proposals = [proposals]

            results, query_feat = self.roi_heads(all_images, features, None, None, proposals, None, None, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
        last_results = [[results[0][-1]]]
        results = [results[0][:-1]]
        
        ## match in the fisrt frame
        if last_box is not None and len(last_box[0][-1]) > 0 and len(results[0][0].pred_boxes) > 0:
            id_label = last_box.id_label

            match_quality_matrix = pairwise_iou(last_box.proposal_boxes, results[0][0].pred_boxes)
            matched_vals, matches = match_quality_matrix.max(dim=0)
            for cnt, cur_iou in enumerate(matched_vals):
                if cur_iou > 0.5:
                    id_0 = results[0][0].id_label[cnt]
                    results[0][0].id_label[cnt] = id_label[int(matches[cnt].data.cpu().numpy())]
                    for cnt_1, item in enumerate(results[0][1].id_label):
                        if item == id_0:
                            results[0][1].id_label[cnt_1] = id_label[int(matches[cnt].data.cpu().numpy())]

        query_result = Instances(results[0][0].image_size)
        query_result.pred_boxes = Boxes(results[0][0].pred_boxes.tensor.detach().clone())
        query_result.scores = results[0][0].scores
        query_result.pred_classes = results[0][0].pred_classes
        query_result.id_label = results[0][0].id_label

        if do_postprocess:
            return TpnRCNN._postprocess(results, batched_inputs, images.image_sizes), last_feat, last_results, query_feat, query_result
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        #print(batched_inputs)
        images = [x["image"].to(self.device) for x in batched_inputs]
        pair_images = [x["pair_image"].to(self.device) for x in batched_inputs]
        if not self.training:
            assert len(batched_inputs) == 1

            all_images = images + pair_images
            all_images = [(x - self.pixel_mean) / self.pixel_std for x in all_images]
            all_images = ImageList.from_tensors(all_images, self.backbone.size_divisibility)

            if batched_inputs[0]["last_feat"] is None:
                images = images + pair_images
            else:
                images = pair_images

            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)

            return images, all_images
        else:
            images = images + pair_images
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)

            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

            # query crop images
            crop_images = [x['crop_images'].to(self.device) for x in batched_inputs]
            crop_images = [(x - self.pixel_mean) / self.pixel_std for x in crop_images]
            crop_images = ImageList.from_tensors(crop_images, self.backbone.size_divisibility)

            return images, support_images, crop_images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []

        instances = instances[0]

        if len(image_sizes) == 1: 
            image_sizes = image_sizes + image_sizes

        batched_inputs = batched_inputs + batched_inputs

        assert len(instances) == len(batched_inputs) == len(image_sizes)
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

