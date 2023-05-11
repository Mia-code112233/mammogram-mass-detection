#@title mask_rcnn
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
from utils import get_datasets
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import torch
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor
import random
import cv2
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer, ColorMode

def visualize_results(dataset_dicts, test_metadata):
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8, 
                    #remove the colors of unsegmented pixels
                    instance_mode=ColorMode.IMAGE_BW    
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        dir_path = "/content/prediction_test"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        basename = os.path.basename(d["file_name"])
        cv2.imwrite(os.path.join(dir_path, basename), v.get_image()[:, :, ::-1])
        #cv2_imshow(v.get_image()[:, :, ::-1])


nums_data = None #set the num of dataset, None means all
dataset_path = "/content/drive/MyDrive/graduation_project/dataset/CBIS_DDSM"


DatasetCatalog.clear()
for d in ["train", "test"]:
    DatasetCatalog.register(d, lambda d = d: get_datasets(nums_data, dataset_path, d, 1))
    MetadataCatalog.get(d).set(thing_classes=["mass"])
    
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   

#set the cfg
cfg = get_cfg()
cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train",)
# no metrics implemented for this dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# only has one class
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
#cfg.MODEL.DEVICE = "cpu"

#set the trainer
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
print(cfg.dump())  


# load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# Set training data-set path
cfg.DATASETS.TEST = ("test", )
# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)

#visual it to have a test
test_metadata = MetadataCatalog.get("test")
dataset_dicts = get_datasets(nums_data, dataset_path, d)
visualize_results(dataset_dicts, test_metadata)


evaluator = COCOEvaluator("test", cfg, False, output_dir="./output/", use_fast_impl=False) #originally my_dataset_test
val_loader = build_detection_test_loader(cfg, "test") #originally my_dataset_test
print(inference_on_dataset(predictor.model, val_loader, evaluator))


'''
model to choose:
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml")
#cfg.merge_from_file("/content/detectron2/configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")

#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detectoon/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detectoon/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detectoon/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl"  # initialize from model zoo
'''