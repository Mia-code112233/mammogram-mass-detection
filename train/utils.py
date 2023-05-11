import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode
import itertools

#get_datasets: to load mass datasets for training
#If clean_flag = 1 it means it use processed datasets for training
def get_datasets(nums_data, dataset_path, data_type, clean_flag = 0):
    json_file = os.path.join(dataset_path, f"annotations/mass_{data_type}_instance.json")
    #json_file = "/Users/cuddly/Desktop/temp/internship/breast_cancer_detection/dataset/CBIS_DDSM/annotations/mass_train_instance.json"
    with open(json_file) as f:
        imgs_anns = json.load(f)
    images_data = imgs_anns['images']
    annotations = imgs_anns['annotations']
    dataset_dicts = []
    #condition 1: use clean datasets for training
    if clean_flag == 1:
        data_dir = os.path.join(dataset_path, f"mass_{data_type}/clean_mass_{data_type}")
    #condition 2: use raw datasets for training
    elif clean_flag == 0 :
        data_dir = os.path.join(dataset_path, f"mass_{data_type}/mass_{data_type}")
    if(nums_data == None):
        nums_data = len(annotations)
    print('test in utils len(images_data)', len(images_data))
    print('test in utils len(annotations)', len(annotations))
    for i in range(nums_data):
        image_id = annotations[i]["image_id"]
        #print('test2 ', len(dataset_dicts), image_id)
        if( image_id > len(dataset_dicts)):
            record = {}
            file_name = os.path.join( data_dir, images_data[image_id-1]['file_name']) 
            height = images_data[image_id-1]['height']
            width = images_data[image_id-1]['width']
            record["file_name"] = file_name
            record["height"] = height
            record["width"] = width
            #record["image_id"] = images_data[i]["id"]
            record["image_id"] = image_id
            objs = []
            record["annotations"] = objs
            dataset_dicts.append(record)
        obj = {
            "bbox": annotations[i]["bbox"],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": annotations[i]["segmentation"],
            "category_id": annotations[i]["category_id"] - 1,
            "iscrowd": annotations[i]["iscrowd"]
        }
        dataset_dicts[image_id - 1]["annotations"].append(obj)
    return dataset_dicts




# write a function that loads the dataset into detectron2's standard format
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    id = 0
    for _, v in imgs_anns.items():
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = id
        id += 1
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = list(itertools.chain.from_iterable(poly))
            print('test in utils', poly)
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

    """
    1, think how to process image_id: 276
    which means len(images_data) != len(annotations)
    """