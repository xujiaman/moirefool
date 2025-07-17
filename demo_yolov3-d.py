from dataset_utils.preprocessing import letterbox_image_padded
from misc_utils.visualization import visualize_detections
from keras import backend as K
from models.yolov3 import YOLOv3_Darknet53,YOLOv3_Darknet53_COCO
from PIL import Image
from tog.attacks import *
import os
import json

import cv2
from random import choice
from attack_utils.moire import parallel_line_moire, rotated_line_moire, curved_line_moire, Moire
from attack_utils.cloth_segmentation import main_generate_mask, load_seg_model
from sklearn.preprocessing import Binarizer
from attack_utils.utils import cal_detection_mAP, append_json_result

K.clear_session()

num_attack_images = 0 # 0 is unlimited

weights = 'model_weights/YOLOv3_Darknet53.h5'  # TODO: Change this path to the victim model's weights

model_type = 'yolo'

detector = YOLOv3_Darknet53(weights=weights)

dataset = 'cocoperson' #cocoperson or ocuhuman

attack_strategy = 'vanishing' # vanishing / fabrication /mislabeling

eps = 80 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 10 / 255.  # Hyperparameter: attack learning rate
n_iter = 20          # Hyperparameter: number of attack iterations

clean_results_file = './results/clean_results.json'
attack_results_file = './results/attack_results.json'


if dataset == 'ocuhuman':
    fpath = ''    # TODO: Change this path to the image to be attacked
    VAL_ANNOTATIONS = ''
else:
    fpath = ''    # TODO: Change this path to the image to be attacked
    VAL_ANNOTATIONS = ''

#-------------------------------------------------

detection_img_names = []
file2id = dict()
with open(VAL_ANNOTATIONS) as f:
    dic = json.load(f)
    images = dic['images']
    for i in range(len(images)):
        image = images[i]
        file2id[image['file_name']] = image['id']

        detection_img_names.append(image['file_name'])

clean_json_results = []
attack_json_results = []

moire_type_choice = 'random_pattern' # 'parallel_line_moire' 'rotated_line_moire' 'curved_line_moire' 'random_pattern'
# moire_type = choice(['parallel_line_moire', 'rotated_line_moire', 'curved_line_moire'])
# moire_type = 'curved_line_moire'
if moire_type_choice == "random_pattern":
    moire_type = choice(['parallel_line_moire', 'rotated_line_moire', 'curved_line_moire'])
else:
    moire_type = moire_type_choice

root_dir = ''
mask_root_dir = ''
sub_folder_name = "yolov3_d_" + dataset + "_" + attack_strategy + "_" + moire_type_choice
save_dir = os.path.join(root_dir, sub_folder_name)
mask_save_dir = os.path.join(mask_root_dir, sub_folder_name)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created new folder: {save_dir}")

if not os.path.exists(mask_save_dir):
    os.makedirs(mask_save_dir)
    print(f"Created new folder: {mask_save_dir}")

device = 'cpu'

attack_img_count = 0
for img in detection_img_names:

    if num_attack_images > 0 and attack_img_count > num_attack_images:
        break

    attack_img_count += 1
    print("attacking {} / {} images".format(attack_img_count, len(detection_img_names)))

    img_path = os.path.join(fpath, img)

    input_img = Image.open(img_path)

    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    # detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
    detections_query = detector.detect(x_query, conf_threshold=0.2)

    # detect cloth area
    detect_boxes = detections_query[:, -4:]

    cloth_masks = np.zeros([1, input_img.size[1], input_img.size[0]])

    seg_model = load_seg_model('', device=device)

    for i in range(len(detect_boxes)):
        if detections_query[i][0] != 14:
            continue

        im_point = np.array(detect_boxes[i]).astype(int)

        xmin = max(int(im_point[-4] * input_img.size[0] / detector.model_img_size[0]), 0)
        ymin = max(int(im_point[-3] * input_img.size[1] / detector.model_img_size[1]), 0)
        xmax = min(int(im_point[-2] * input_img.size[0] / detector.model_img_size[0]), input_img.size[0])
        ymax = min(int(im_point[-1] * input_img.size[1] / detector.model_img_size[1]), input_img.size[1])

        reloc_box = [xmin, ymin, xmax, ymax]
        im_p = input_img.crop(np.array(reloc_box))

        _, mask = main_generate_mask(im_p, seg_model, device)

        cloth_masks[0][ymin:ymax, xmin:xmax] += mask[0]

    cloth_masks = np.clip(cloth_masks, 0, 1)

    # -----------------------------------------------------------------------
    clean_json_results = append_json_result(clean_json_results, detections_query, x_meta, input_img, file2id, img,
                                            cloth_masks, model_type)

    # np.save(mask_save_dir + "/" + img[:-4], cloth_masks)
    np.savez_compressed(mask_save_dir + "/" + img[:-4], cloth_masks=cloth_masks)
    # -------------------------------------------------

    # generate moire mask
    moire = Moire()


    im0 = cv2.imread(img_path)

    moire_img, moire_mask = moire(cloth_masks, im0, moire_type)
    im0 = np.clip(moire_img * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(save_path, im0)
    binarizer = Binarizer(threshold=0.2)
    bi_moire_mask = binarizer.transform(moire_mask)

    moire_mask = Image.fromarray(bi_moire_mask * 255)
    mask_query, _ = letterbox_image_padded(moire_mask, size=detector.model_img_size)
    fi_moire_mask = mask_query[0, :, :, 0]
    fi_moire_mask = binarizer.transform(fi_moire_mask)

    if attack_strategy == 'vanishing':
        x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter,
                                        mask=fi_moire_mask)
        detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)

        attack_detections_query = detections_adv_vanishing

        img_save = cv2.cvtColor((x_adv_vanishing[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)


    if attack_strategy == 'fabrication':
        x_adv_fabrication = tog_fabrication(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter,
                                            mask=fi_moire_mask)

        detections_adv_fabrication = detector.detect(x_adv_fabrication,
                                                     conf_threshold=detector.confidence_thresh_default)

        attack_detections_query = detections_adv_fabrication

        img_save = cv2.cvtColor((x_adv_fabrication[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)


    if attack_strategy == 'mislabeling':
        x_adv_mislabeling_ml = tog_mislabeling(victim=detector, x_query=x_query, target='ml', n_iter=n_iter, eps=eps,
                                               eps_iter=eps_iter, mask=fi_moire_mask)
        x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=n_iter, eps=eps,
                                               eps_iter=eps_iter, mask=fi_moire_mask)

        detections_adv_mislabeling_ml = detector.detect(x_adv_mislabeling_ml,
                                                        conf_threshold=detector.confidence_thresh_default)
        detections_adv_mislabeling_ll = detector.detect(x_adv_mislabeling_ll,
                                                        conf_threshold=detector.confidence_thresh_default)


        attack_detections_query = detections_adv_mislabeling_ml

        img_save = cv2.cvtColor((x_adv_mislabeling_ml[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # save the adversarial images
    img_save_true = img_save[x_meta[1]:(detector.model_img_size[1]-x_meta[1]),x_meta[0]:(detector.model_img_size[0]-x_meta[0]),:]
    cv2.imwrite(save_dir+"/"+img[:-4]+'.png', img_save_true, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    if len(attack_detections_query)!= 0:
        attack_json_results = append_json_result(attack_json_results, attack_detections_query, x_meta, input_img, file2id,
                                             img, cloth_masks, model_type)

with open(clean_results_file, 'w') as f:
    f.write(json.dumps(clean_json_results, indent=4))

with open(attack_results_file, 'w') as f:
    f.write(json.dumps(attack_json_results, indent=4))

clean_mAP = cal_detection_mAP(VAL_ANNOTATIONS, clean_results_file)
attack_mAP = cal_detection_mAP(VAL_ANNOTATIONS, attack_results_file)
ASR = (clean_mAP - attack_mAP) / clean_mAP

print('mAP (clean): {}'.format(clean_mAP))
print('mAP (attack): {}'.format(attack_mAP))
print('attack success rate: ' + str(ASR))