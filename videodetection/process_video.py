import cv2
import numpy as np
from visualize_cv2 import model, display_instances, class_names
import tensorflow as tf
import os
import sys
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = sys.argv
if(len(args) < 2):
	print("run command: python video_demo.py 0 or video file name")
	sys.exit(0)
name = args[1]
if(len(args[1]) == 1):
	name = int(args[1])
	
stream = cv2.VideoCapture(name)
width, height = stream.get(cv2.CAP_PROP_FRAME_WIDTH), stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
out = cv2.VideoWriter('video_out.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (int(width),int(height)))
while True:
	ret , frame = stream.read()
	
	results = model.detect([frame], verbose=1)

	# Visualize results
	r = results[0]
	masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
	cv2.namedWindow("masked_image", cv2.WINDOW_NORMAL)
	cv2.imshow("masked_image",masked_image)
	out.write(masked_image)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
stream.release()
cv2.destroyWindow("masked_image")
