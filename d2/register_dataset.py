# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import os
import json
from tqdm import tqdm
from detectron2.data import DatasetCatalog
from detectron2.structures import BoxMode


CATEGORIES = [
      "header",
      "question-answer",
      "question",
      "question_label",
      "A",
      "A_label",
      "B",
      "B_label",
      "C",
      "C_label",
      "D",
      "D_label"
  ]

CATEGORY_TO_ID = dict([(cat, id) for id, cat in enumerate(CATEGORIES)] )

def main():
  def get_all_anno(value, key, scale):
    res = []
    if type(value) == list:
      if type(value[0]) == int or type(value[0]) == float:
        if value[2] == value[0] or value[3] == value[1]:
          return []
        res.append({"bbox": list(scale * np.array(value, dtype=np.float)),
                  "bbox_mode": BoxMode.XYXY_ABS,
                  # "bbox_mode": 0,
                  "category_id": CATEGORY_TO_ID[key],
                  "category": key,
                  "segmentation": []})
      else: # list of dict
        for d in value:
          res += get_all_anno(d, None, scale)
    elif type(value) == dict:
      for k,v in value.items():
        res += get_all_anno(v, k, scale)
    else:
      print("value is not dict or list")
      exit(1)
    return res

  def parse_annotation(anno_file, scale):
    with open(anno_file) as f:
      raw = json.load(f)
    parsed = get_all_anno(raw, None, scale)
    return parsed

  def create_coco_dict(id_dir, img_id, height, width, scale):
    if "img.png" not in os.listdir(id_dir): 
      return None
    res = {}
    res["file_name"] = os.path.join(id_dir, "img.png")
    res["height"] = height
    res["width"] = width
    res["image_id"] = img_id 

    anno_file = os.path.join(id_dir, "anno.json")
    res["annotations"] = parse_annotation(anno_file, scale)
    return res


  def my_dataset_function(data_dir, start, end):
    # data_dir = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/idris_samples"
    res = []
    ids = sorted(os.listdir(data_dir))
    N = len(ids)
    start_id = int(start*N)
    end_id = int(end*N)
    for sample_id in tqdm(ids[start_id:end_id]):
      if sample_id == ".DS_Store": continue
      id_dir = os.path.join(data_dir, sample_id)
      coco_dict = create_coco_dict(id_dir, sample_id, 1414, 1000, 2) 
      if coco_dict is not None:
        res.append(coco_dict)
    return res

  def create_coco_dict_with_prefix(id_dir, img_id, height, width, scale, prefix):
    if "img.png" not in os.listdir(id_dir): 
      return None
    res = {}
    res["file_name"] = os.path.join(prefix, "img.png")
    res["height"] = height
    res["width"] = width
    res["image_id"] = img_id 

    anno_file = os.path.join(id_dir, "anno.json")
    res["annotations"] = parse_annotation(anno_file, scale)
    return res

  def merge_to_one_json(data_dir, start, end, data_prefix=""):
    # data_dir = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/idris_samples"
    res = []
    ids = sorted(os.listdir(data_dir))
    N = len(ids)
    start_id = int(start*N)
    end_id = int(end*N)
    for sample_id in tqdm(ids[start_id:end_id]):
      if sample_id == ".DS_Store": continue
      id_dir = os.path.join(data_dir, sample_id)
      coco_dict = create_coco_dict_with_prefix(id_dir, sample_id, 1414, 1000, 2, os.path.join(data_prefix,sample_id)) 
      if coco_dict is not None:
        res.append(coco_dict)
    return res

  def anno_from_json(json_file, data_dir):
    with open(json_file) as f:
      anno = json.load(f)
    for obj in anno:
      obj["file_name"] = os.path.join(data_dir, obj["file_name"])
      for a in obj["annotations"]:
        if a["bbox_mode"] == 0:
          a["bbox_mode"] = BoxMode.XYXY_ABS
        else:
          raise ValueError(f"bbox_mode {a['bbox_mode']} not supported")
    return anno

  data_dir = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/detection_data_with_qa_labels"
  # data_dir = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/idris_samples"

  train_json = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/anno/train_anno.json"
  val_json = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/anno/val_anno.json"
  test_json = "/content/drive/MyDrive/Colab Notebooks/idris-teacher-assistant/anno/test_anno.json"

  DatasetCatalog.register("my_dataset_val", lambda: anno_from_json(val_json, data_dir))
  DatasetCatalog.register("my_dataset_train", lambda: anno_from_json(train_json, data_dir))
  DatasetCatalog.register("my_dataset_test", lambda: anno_from_json(test_json, data_dir))

  from detectron2.data import MetadataCatalog
  MetadataCatalog.get("my_dataset_train").thing_classes = CATEGORIES
  MetadataCatalog.get("my_dataset_val").thing_classes = CATEGORIES
  MetadataCatalog.get("my_dataset_test").thing_classes = CATEGORIES

  #visualize training data
  my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
  dataset_dicts = DatasetCatalog.get("my_dataset_train")

  for d in random.sample(dataset_dicts, 3):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])

if __name__ == "__main__":
  main()