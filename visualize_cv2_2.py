import cv2
import numpy as np
import os
import sys
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from samples.coco.coco import CocoConfig
from imutils.video import WebcamVideoStream
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import random
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT_DIR = os.path.abspath("/home/ben/Mask_RCNN")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)



class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}
'''
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
'''
def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image
'''
def display_instances(image, boxes, masks, class_ids, class_names, scores, colors=None):
    N = boxes.shape[0]
    #if not N:
    #    print('\n*** No instances to display *** \n')
    #else:
    #    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    if N:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    colors = colors
    # colors = random_colors(N)
    masked_image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    for i in range(N):
        class_id = class_ids[i]
        color = colors[class_id-1]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        camera_color = (color[0] * 255, color[1] * 255, color[2] * 255)
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), camera_color, 1)

        # Mask
        mask = masks[:, :, i]
        alpha = 0.5
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1,
                                             image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                             masked_image[:, :, c])

        # Label
        score = scores[i]
        label = class_names[class_id]
        caption = '{} {:.2f}'.format(label, score) if score else label

        # Get caption text size
        ret, baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Put the rectangle and text on the top left corner of the bounding box
        cv2.rectangle(masked_image, (x1, y1), (x1 + ret[0], y1 + ret[1] + baseline), camera_color, -1)
        cv2.putText(masked_image, caption, (x1, y1 + ret[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, lineType=cv2.LINE_AA)


        # Put the rectangle and text on the bottom left corner
        # cv2.rectangle(masked_image, (x1, y2 - ret[1] - baseline), (x1 + ret[0], y2), camera_color, -1)
        # cv2.putText(masked_image, caption, (x1, y2 - baseline),
                    # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return masked_image.astype(np.uint8)
'''



def display_instances(image, boxes, masks, ids, names, scores, n1, T):
    """
        take the image and results and apply the mask, box, and Label
    """
    N = boxes.shape[0]
    if not N:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(N):
        if not np.any(boxes[i]):
            continue
               

        y1, x1, y2, x2 = boxes[i]

        #if y2 == 904:
            #n1 = n1	
        #if 907 > y2 >= 900:
            #n1 = n1 + 1
        #if y2 == 896:
            #n1 = n1 + 1
            
        #else:
            #n1 = n1
        if y2 == 891:
            n1 = n1 + 1
        if y2 == 875:
            n1 = n1 + 1
        if y2 == 909:
            n1 = n1 + 1

        label = names[ids[i]] 
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        image = cv2.rectangle(image, (1000, 871), (1300, 891), (0,0,255), 2)
        text = 'total:' + str(n1)
        image = cv2.putText(
            image, text, (0, 500), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), 2
        )

        #text = str(y2)
        #image = cv2.putText(
            #image, text, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        #)

    return image, n1, T


"""
if __name__ == '__main__':
    
        test everything
    

    capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
"""
