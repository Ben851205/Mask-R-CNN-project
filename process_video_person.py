import cv2
import numpy as np
from visualize_cv2_2 import model, display_instances, class_names
import tensorflow as tf
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
n1 = 15
T = 1
capture = cv2.VideoCapture('person_count.mp4')
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'MJPG')
output = cv2.VideoWriter('output_person_phone.mp4', codec, 20.0, size)

while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
       
        r = results[0]
        #print(r['rois'][0])
        if class_names.index('person') in r['class_ids']:
            k = list(np.where(r['class_ids'] == class_names.index('person'))[0])
            r['scores'] = np.array([r['scores'][i] for i in k])
            r['rois'] = np.array([r['rois'][i] for i in k])
            r['masks'] = np.transpose(r['masks'])
            r['masks'] = np.array([r['masks'][i] for i in k])
            r['masks'] = np.transpose(r['masks'])
            r['class_ids'] = np.array([r['class_ids'][i] for i in k])
        frame, n1, T = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], n1, T)
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
output.release()
cv2.destroyAllWindows()
