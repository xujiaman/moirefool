

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


if __name__ == "__main__":
    dataset = 'cocoperson'  # cocoperson or ochuman

    if dataset == 'ocuhuman':
        coco_true = COCO(annotation_file='')        #标注文件的路径及文件名，json文件形式
        coco_pre = coco_true.loadRes( '')
    else:
        coco_true = COCO(annotation_file='')        #标注文件的路径及文件名，json文件形式
        # cocoDt = cocoGt.loadRes('')  #自己的生成的结果的路径及文件名，json文件形式
        coco_pre = coco_true.loadRes( '')

    cocoEval = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()