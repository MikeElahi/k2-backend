import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer


class Detector:
    DEFAULT_MODEL = "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"

    def __init__(self, model: str = DEFAULT_MODEL):
        setup_logger()
        with torch.no_grad():
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(model))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
            self.cfg = cfg
            self.predictor = DefaultPredictor(cfg)

    def predict(self, image):
        with torch.no_grad():
            panoptic_seg, segments_info = self.predictor(image)["panoptic_seg"]
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
            v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
            out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
            return panoptic_seg, segments_info, out.get_image()[:, :, ::-1], metadata
