# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
#from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
#from detectron2.utils.visualizer import ColorMode, Visualizer
from visualizer import ColorMode, Visualizer

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


metadata_fsvod_test = {"thing_classes": ['JetLev-Flyer', 'addax', 'aircraft_carrier', 'airship', 'alligator', 'amphibian', 'anteater', 'antelope', 'aoudad', 'armadillo', 'asian_crocodile', 'autogiro', 'ax', 'baby_buggy', 'bactrian_camel', 'badger', 'balance_car', 'balloon', 'barracuda', 'barrow', 'bathyscaphe', 'belgian_hare', 'berlin', 'bezoar_goat', 'binturong', 'black_leopard', 'black_rabbit', 'black_rhinoceros', 'black_squirrel', 'boar', 'bow_(weapon)', 'brahman', 'bucket', 'cabbageworm', 'camel', 'canada_porcupine', 'cashmere_goat', 'cheetah', 'chiacoan_peccary', 'chimaera', 'chimpanzee', 'chinese_paddlefish', 'civet', 'coin', 'collared_peccary', 'cornetfish', 'corvette', 'crab', 'crayfish', 'cruise_missile', 'deer', 'destroyer_escort', 'dogsled', 'dragon-lion_dance', 'dumpcart', 'earwig', 'eastern_grey_squirrel', 'elasmobranch', 'elk', 'european_hare', 'fall_cankerworm', 'fanaloka', 'fish', 'flag', 'forest_goat', 'fox', 'fox_squirrel', 'garden_centipede', 'gavial', 'gemsbok', 'genet', 'giant_armadillo', 'giant_kangaroo', 'giant_panda', 'goat', 'goral', 'gorilla', 'guanaco', 'guard_ship', 'guitar', 'gun', 'half_track', 'hammerhead', 'hand_truck', 'helicopter', 'hermit_crab', 'hippo', 'hog', 'hog-nosed_skunk', 'horse_cart', 'horseshoe_crab', 'humvee', 'ibex', 'indian_rhinoceros', 'jaguar', 'jinrikisha', 'knitting_needle', 'koala', 'lander', 'langur', 'lappet_caterpillar', 'lemur', 'leopard', 'lesser_kudu', 'lesser_panda', 'lion', 'long-tailed_porcupine', 'luge', 'malayan_tapir', 'manatee', 'mangabey', 'medusa', 'millipede', 'minisub', 'monkey', 'mop', 'mouflon', 'mountain_goat', 'multistage_rocket', 'orangutan', 'oxcart', 'pacific_walrus', 'panzer', 'peba', 'pedicab', 'peludo', "pere_david's_deer", 'piano', 'pistol', 'pony_cart', 'pung', 'rabbit', 'raccoon', 'reconnaissance_vehicle', 'red_squirrel', 'robot', 'rubic_cube', 'sassaby', 'saxophone', 'scraper', 'seal', 'sepia', 'serow', 'shark', 'shawl', 'shopping_cart', 'shrimp', 'skibob', 'sloth', 'small_crocodile', 'snow_leopard', 'snowmobil', 'sow', 'spider_monkey', 'spotted_skunk', 'squirrel', 'suricate', 'swing', 'sword', 'tadpole_shrimp', 'tank', 'tiger', 'tiglon', 'toboggan', 'turtle', 'unicycle', 'urial', 'virginia_deer', 'walking_stick', 'warthog', 'whale', 'wheelchair', 'white-tailed_jackrabbit', 'white_crocodile', 'white_rabbit', 'white_rhinoceros', 'white_squirrel', 'wildboar', 'woolly_monkey', 'yellow-throated_marten']
}

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
        #if len(cfg.DATASETS.TEST):
        #    self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.metadata = metadata_fsvod_test
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, category_id):
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
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width, "annotations": [{"category_id": category_id}]}
            predictions = self.model([inputs])[0]
            return predictions

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        #self.metadata = MetadataCatalog.get(
        #    cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        #)
        self.metadata = metadata_fsvod_test
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, ann, tao):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None

        category_id = ann['category_id']
        category_name = tao.load_cats([category_id])

        predictions = self.predictor(image, category_id)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances, gt_instances=ann)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
