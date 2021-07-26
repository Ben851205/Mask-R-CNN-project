import os
import sys
import cv2
import time
import imutils
import numpy as np
import mrcnn.model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from imutils.video import WebcamVideoStream
import random
import os
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
from samples.coco.coco import CocoConfig

ROOT_DIR = os.path.abspath("/home/ben/Mask_RCNN")

#sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0200.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'Wear', 'None']

colors = visualize.random_colors(len(class_names))

gentle_grey = (45, 65, 79)
white = (255, 255, 255)

OPTIMIZE_CAM = True
SHOW_FPS = False
SHOW_FPS_WO_COUNTER = True # faster
PROCESS_IMG = True


if OPTIMIZE_CAM:
    vs = WebcamVideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(0)

if SHOW_FPS:
    fps_caption = "FPS: 0"
    fps_counter = 0
    start_time = time.time()

SCREEN_NAME = 'Mask RCNN LIVE'
cv2.namedWindow(SCREEN_NAME, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(SCREEN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow(SCREEN_NAME, 640, 480)

while True:
    # Capture frame-by-frame
    if OPTIMIZE_CAM:
        frame = vs.read()
    else:
        grabbed, frame = vs.read()
        if not grabbed:
            break
    
    if SHOW_FPS_WO_COUNTER:
        start_time = time.time() # start time of the loop

    if PROCESS_IMG:
        results = model.detect([frame])
        r = results[0]

        # Run detection
        masked_image = visualize.display_instances_10fps(frame, r['rois'], r['masks'], 
            r['class_ids'], class_names, r['scores'], colors=colors, real_time=True)
        
    if PROCESS_IMG:
        s = masked_image
    else:
        s = frame
    # print("Image shape: {1}x{0}".format(s.shape[0], s.shape[1]))

    width = s.shape[1]
    height = s.shape[0]
    top_left_corner = (width-120, height-20)
    bott_right_corner = (width, height)
    top_left_corner_cvtext = (width-80, height-5)

    if SHOW_FPS:
        fps_counter+=1
        if (time.time() - start_time) > 5 : # every 5 second
            fps_caption = "FPS: {:.0f}".format(fps_counter / (time.time() - start_time))
            # print(fps_caption)
            fps_counter = 0
            start_time = time.time()
        ret, baseline = cv2.getTextSize(fps_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(s, (width - ret[0], height - ret[1] - baseline), bott_right_corner, gentle_grey, -1)
        cv2.putText(s,fps_caption, (width - ret[0], height - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, lineType=cv2.LINE_AA)

    if SHOW_FPS_WO_COUNTER:
        # Display the resulting frame
        fps_caption = "FPS: {:.0f}".format(1.0 / (time.time() - start_time))
        # print("FPS: ", 1.0 / (time.time() - start_time))
        
        # Put the rectangle and text on the bottom left corner
        ret, baseline = cv2.getTextSize(fps_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(s, (width - ret[0], height - ret[1] - baseline), bott_right_corner, gentle_grey, -1)
        cv2.putText(s, fps_caption, (width - ret[0], height - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, lineType=cv2.LINE_AA)

    
    s = cv2.resize(s,(640,480))
    cv2.imshow(SCREEN_NAME, s)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
if OPTIMIZE_CAM:
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()

