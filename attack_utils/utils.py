import os.path as osp
import os
import numpy as np


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def cal_detection_mAP(clean_result_json, attck_result_json):
    # dataset = 'cocoperson'  # cocoperson or ochuman


    coco_true = COCO(annotation_file=clean_result_json)        #标注文件的路径及文件名，json文件形式
    coco_pre = coco_true.loadRes(attck_result_json)

    cocoEval = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP = cocoEval.stats[5]

    return mAP

def append_json_result(json_results, detections_query, x_meta, input_img, file2id, img, cloth_masks, model_type):
    image_id = file2id[img]
    clss = detections_query[:, 0]
    confs = detections_query[:, 1]
    boxes = detections_query[:, -4:]
    for box, conf, cls in zip(boxes, confs, clss):
        xmin = max(int((box[-4] - x_meta[0]) / x_meta[4]), 0)
        ymin = max(int((box[-3] - x_meta[1]) / x_meta[4]), 0)
        xmax = min(int((box[-2] - x_meta[0]) / x_meta[4]), input_img.size[0])
        ymax = min(int((box[-1] - x_meta[1]) / x_meta[4]), input_img.size[1])

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        cloth_detect = True if np.sum(cloth_masks[0, ymin:ymax, xmin:xmax])/(w * h) > 0.04 else False

        if cloth_detect == True:
            if cls == 14 and (model_type == 'yolo' or model_type == 'faster_rcnn'):
                cls = 1

            if cls == 15 and model_type == 'ssd' :
                cls = 1

            json_results.append({'area': w * h,
                                 'image_id': image_id,
                                 'category_id': np.int(cls),
                                 'iscrowd': 0,
                                 'id': image_id,
                                 'ignore': 0,
                                 'segmentation': [],
                                 'bbox': [x, y, w, h],
                                 'score': np.float(conf)})
    return json_results