import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names
import tensorflow as tf
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

capture = cv2.VideoCapture('Mask0319_phone.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('output_mask_phone.mp4', codec, 20.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=1)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
output.release()
cv2.destroyAllWindows()
