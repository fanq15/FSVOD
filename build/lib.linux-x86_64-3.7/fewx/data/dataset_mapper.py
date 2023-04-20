# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pandas as pd
from detectron2.data.catalog import MetadataCatalog
import os

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithSupport"]


class DatasetMapperWithSupport:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.few_shot       = cfg.INPUT.FS.FEW_SHOT
        self.video_flag     = cfg.INPUT.FS.VIDEO_FLAG
        self.support_way       = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot       = cfg.INPUT.FS.SUPPORT_SHOT
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train
        self.tpn_flag = True

        if self.is_train:
            # support_df
            self.support_on = True
            if self.few_shot:
                if self.video_flag:
                    pkl_path = "./datasets/fsvod/10_shot_support_df.pkl"
                    assert False
                else:
                    pkl_path = "./datasets/coco/10_shot_support_df.pkl"
            else:
                if self.video_flag:
                    pkl_path = "./datasets/fsvod/fsvod_train_support_df.pkl"
                    ori_pkl_path = "./datasets/fsvod/fsvod_train_df.pkl"
                    self.ori_support_df = pd.read_pickle(ori_pkl_path)
                else:
                    pkl_path = "./datasets/coco/train_support_df.pkl"
            
            self.support_df = pd.read_pickle(pkl_path)
            if not self.video_flag:
                metadata = MetadataCatalog.get('coco_2017_train')
                # unmap the category mapping ids for COCO
                reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
                self.support_df['category_id'] = self.support_df['category_id'].map(reverse_id_mapper)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.is_train:
            # support
            if self.support_on:
                if "annotations" in dataset_dict:
                    # USER: Modify this if you want to keep them for some reason.
                    for anno in dataset_dict["annotations"]:
                        if not self.mask_on:
                            anno.pop("segmentation", None)
                        if not self.keypoint_on:
                            anno.pop("keypoints", None)
                support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
                dataset_dict['support_images'] = torch.as_tensor(np.ascontiguousarray(support_images))
                dataset_dict['support_bboxes'] = support_bboxes
                dataset_dict['support_cls'] = support_cls

        # generate the next frame or former frame
        if self.tpn_flag:
            target_anno = dataset_dict['annotations'][0]

            ann_id = target_anno['id']
            track_id = target_anno['track_id']
            video_id = target_anno['video_id']
            target_category_id = target_anno['category_id']

            id_ls = self.ori_support_df.loc[(self.ori_support_df['category_id'] == target_category_id) & (self.ori_support_df['track_id']==track_id), 'id'].tolist()
            id_ls_len = len(id_ls)
            assert len(set(id_ls)) == id_ls_len

            ls_idx = id_ls.index(ann_id)
            if ls_idx == 0:
                pair_index = ls_idx + 1
            elif ls_idx == id_ls_len - 1:
                pair_index = ls_idx - 1
            else:
                np.random.seed(ann_id)
                next_flag = np.random.choice(2, 1)[0] # 0 or 1
                if next_flag:
                    pair_index = ls_idx + 1
                else:
                    pair_index = ls_idx - 1

            pair_id = id_ls[pair_index]

            pair_df = self.ori_support_df.loc[(self.ori_support_df['category_id'] == target_category_id) & (self.ori_support_df['id'] == pair_id), :]
            pair_image_id = pair_df['image_id'].tolist()[0]

            pair_df = self.ori_support_df.loc[(self.ori_support_df['category_id'] == target_category_id) & (self.ori_support_df['image_id'] == pair_image_id), :]
            assert len(set(pair_df['image_id'].tolist())) == 1 # one image contains multiple instances
            
            root_path = './datasets/fsvod/images'
            pair_img = utils.read_image(os.path.join(root_path, pair_df['file_name'].tolist()[0]), format=self.img_format) # one image contains multiple instances
            utils.check_image_size(dataset_dict, pair_img)

            # pair annotations
            pair_annotations = []
            for item_id in range(pair_df.shape[0]):
                iscrowd = 0
                bbox = pair_df['bbox'].tolist()[item_id]
                category_id = pair_df['category_id'].tolist()[item_id]
                id = pair_df['id'].tolist()[item_id]
                track_id = pair_df['track_id'].tolist()[item_id]
                video_id = pair_df['video_id'].tolist()[item_id]
                bbox_mode = dataset_dict['annotations'][0]['bbox_mode']
                item_annotations = {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'id': id, 'track_id': track_id, 'video_id': video_id, 'bbox_mode': bbox_mode}
                pair_annotations.append(item_annotations)

            # for original annotations, we need to add other cls's annotations.
            # because we split the image into different cls annotations.
            cur_df = self.ori_support_df.loc[(self.ori_support_df['category_id'] == target_category_id) & (self.ori_support_df['id'] == ann_id), :]
            cur_image_id = cur_df['image_id'].tolist()[0]

            cur_df = self.ori_support_df.loc[(self.ori_support_df['category_id'] == target_category_id) & (self.ori_support_df['image_id'] == cur_image_id), :]
            assert len(set(cur_df['image_id'].tolist())) == 1 # one image contains multiple instances

            # cur annotations
            cur_annotations = []
            for item_id in range(cur_df.shape[0]):
                iscrowd = 0
                bbox = cur_df['bbox'].tolist()[item_id]
                category_id = cur_df['category_id'].tolist()[item_id]
                id = cur_df['id'].tolist()[item_id]
                track_id = cur_df['track_id'].tolist()[item_id]
                video_id = cur_df['video_id'].tolist()[item_id]
                bbox_mode = dataset_dict['annotations'][0]['bbox_mode']
                cur_item_annotations = {'iscrowd': iscrowd, 'bbox': bbox, 'category_id': category_id, 'id': id, 'track_id': track_id, 'video_id': video_id, 'bbox_mode': bbox_mode}
                cur_annotations.append(cur_item_annotations)

            dataset_dict['annotations'] = cur_annotations

            # cur image query crops
            crop_images, crop_bboxes, crop_cls = self.generate_query_crop(cur_annotations)
            dataset_dict['crop_images'] = torch.as_tensor(np.ascontiguousarray(crop_images))
            dataset_dict['crop_bboxes'] = crop_bboxes
            dataset_dict['crop_cls'] = crop_cls

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                assert Flase # we do not support crop currently
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            track_id_ls = []
            for item in dataset_dict['annotations']:
                track_id_ls.append(item['track_id'])

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            instances.gt_tracks = torch.tensor(track_id_ls, dtype=torch.int64)

            if self.tpn_flag:
                # pair annotations processing
                pair_annos = [
                    utils.transform_instance_annotations(
                        obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                    )
                    for obj in pair_annotations
                    if obj.get("iscrowd", 0) == 0
                ]
                pair_instances = utils.annotations_to_instances(
                    pair_annos, image_shape, mask_format=self.mask_format
                )

                track_id_ls = []
                for item in pair_annotations:
                    track_id_ls.append(item['track_id'])
                pair_instances.gt_tracks = torch.tensor(track_id_ls, dtype=torch.int64)

                pair_img = transforms.apply_image(pair_img)
                dataset_dict["pair_image"] = torch.as_tensor(np.ascontiguousarray(pair_img.transpose(2, 0, 1)))
                dataset_dict['pair_instances'] = utils.filter_empty_instances(pair_instances)

            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

    def generate_support(self, dataset_dict):
        #support_way = self.support_way #2
        #support_shot = self.support_shot #5
        support_way = 3
        support_shot = 5

        id = dataset_dict['annotations'][0]['id']
        query_cls = self.ori_support_df.loc[self.ori_support_df['id']==id, 'category_id'].tolist()[0] # they share the same category_id and image_id
        query_img = self.ori_support_df.loc[self.ori_support_df['id']==id, 'image_id'].tolist()[0]
        all_cls = self.ori_support_df.loc[self.ori_support_df['image_id']==query_img, 'category_id'].tolist()

        if self.video_flag:
            query_video = self.ori_support_df.loc[self.ori_support_df['id']==id, 'video_id'].tolist()[0]

        # Crop support data and get new support box in the support data
        support_data_all = np.zeros((support_way * support_shot + 5, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((support_way * support_shot + 5, 4), dtype = np.float32)
        used_image_id = [query_img]

        used_id_ls = []
        for item in dataset_dict['annotations']:
            used_id_ls.append(item['id'])
        #used_category_id = [query_cls]
        used_category_id = list(set(all_cls))
        support_category_id = []
        mixup_i = 0

        for shot in range(support_shot * 2):
            # Support image and box
            if self.video_flag:
                try:
                    support_id = self.support_df.loc[(self.support_df['category_id'] == query_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls) & (self.support_df['video_id'] != query_video)), 'id'].sample(random_state=id).tolist()[0]
                except:
                    support_id = self.support_df.loc[(self.support_df['category_id'] == query_cls) & (self.support_df['video_id'] != query_video), 'id'].sample(random_state=id).tolist()[0]
            else:
                support_id = self.support_df.loc[(self.support_df['category_id'] == query_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls)), 'id'].sample(random_state=id).tolist()[0]
            support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
            support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]
            used_id_ls.append(support_id) 
            used_image_id.append(support_img)

            support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
            assert support_db['id'].values[0] == support_id
            
            if self.video_flag: 
                support_data = utils.read_image('./datasets/fsvod/' + support_db["file_path"].tolist()[0], format=self.img_format)
            else:
                support_data = utils.read_image('./datasets/coco/' + support_db["file_path"].tolist()[0], format=self.img_format)
            support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
            support_box = support_db['support_box'].tolist()[0]
            #print(support_data)
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box
            support_category_id.append(support_cls)
            mixup_i += 1

        if support_way == 1:
            pass
        else:
            for way in range(support_way-1):
                other_cls = self.support_df.loc[(~self.support_df['category_id'].isin(used_category_id)), 'category_id'].drop_duplicates().sample(random_state=id).tolist()[0]
                used_category_id.append(other_cls)
                for shot in range(support_shot):
                    # Support image and box
                    if self.video_flag:
                        #print(self.support_df.loc[(self.support_df['category_id'] == query_cls) & (self.support_df['video_id'] != query_video), 'video_id'])
                        support_id = self.support_df.loc[(self.support_df['category_id'] == other_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls) & (self.support_df['video_id'] != query_video)), 'id'].sample(random_state=id).tolist()[0]
                    else:
                        support_id = self.support_df.loc[(self.support_df['category_id'] == other_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls)), 'id'].sample(random_state=id).tolist()[0]
                     
                    support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
                    support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]

                    used_id_ls.append(support_id) 
                    used_image_id.append(support_img)

                    support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
                    assert support_db['id'].values[0] == support_id

                    if self.video_flag: 
                        support_data = utils.read_image('./datasets/fsvod/' + support_db["file_path"].tolist()[0], format=self.img_format)
                    else:
                        support_data = utils.read_image('./datasets/coco/' + support_db["file_path"].tolist()[0], format=self.img_format)

                    support_box = support_db['support_box'].tolist()[0]
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all[mixup_i] = support_data
                    support_box_all[mixup_i] = support_box
                    support_category_id.append(support_cls)
                    mixup_i += 1
        
        return support_data_all, support_box_all, support_category_id

    def generate_query_crop(self, annotations):
        id = annotations[0]['id']
        query_cls = self.ori_support_df.loc[self.ori_support_df['id']==id, 'category_id'].tolist()[0] # they share the same category_id and image_id
        query_img = self.ori_support_df.loc[self.ori_support_df['id']==id, 'image_id'].tolist()[0]
        all_cls = self.ori_support_df.loc[self.ori_support_df['image_id']==query_img, 'category_id'].tolist()

        query_video = self.ori_support_df.loc[self.ori_support_df['id']==id, 'video_id'].tolist()[0]
        query_track = self.ori_support_df.loc[self.ori_support_df['id']==id, 'track_id'].tolist()[0]

        total_num = self.support_df.loc[self.support_df['track_id'] == query_track].shape[0]
        total_num = min(total_num, 10)
        np.random.seed(id)
        #query_shot = np.random.randint(2, total_num+1)
        query_shot = total_num

        # Crop support data and get new support box in the support data
        support_data_all = np.zeros((query_shot, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((query_shot, 4), dtype = np.float32)
        #used_image_id = [query_img]
        used_image_id = []

        used_id_ls = []
        #for item in annotations:
        #    used_id_ls.append(item['id'])
        #used_category_id = [query_cls]
        used_category_id = list(set(all_cls))
        support_category_id = []
        mixup_i = 0
        for shot in range(query_shot):
            # Support image and box
            support_id = self.support_df.loc[(self.support_df['category_id'] == query_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls) & (self.support_df['video_id'] == query_video) & (self.support_df['track_id'] == query_track)), 'id'].sample(random_state=id).tolist()[0]
            support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
            support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]
            used_id_ls.append(support_id) 
            used_image_id.append(support_img)

            support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
            assert support_db['id'].values[0] == support_id
            
            support_data = utils.read_image('./datasets/fsvod/' + support_db["file_path"].tolist()[0], format=self.img_format)
            support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
            support_box = support_db['support_box'].tolist()[0]

            # disturb the support_box
            low = -0.15
            high = 0.15 
            rand_ratio = np.random.uniform(low, high, 4)
            box_height = support_box[2] - support_box[0]
            box_width = support_box[3] - support_box[1]

            support_box[0] += rand_ratio[0] * box_height
            support_box[1] += rand_ratio[1] * box_width
            support_box[2] += rand_ratio[2] * box_height
            support_box[3] += rand_ratio[3] * box_width

            #print(support_data)
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box
            support_category_id.append(0) #support_cls)
            mixup_i += 1

        support_box_all = np.clip(support_box_all, 0., 320.)

        return support_data_all, support_box_all, support_category_id
