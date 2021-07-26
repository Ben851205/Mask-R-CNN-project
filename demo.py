
#Mask R-CNN例子
#一个使用预训练模型检测和分割目标的简要介绍

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ ['TF_FORCE_GPU_ALLOW_GROWTH'] ='true'


# Root directory of the project
#项目根目录（返回上级目录的绝对路径）
ROOT_DIR = os.path.abspath("/home/ben/Mask_RCNN")


# Import Mask RCNN
#导入Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library为了找到本地版本的库
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
#引用COCO设置模块
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version（即将coco模块的目录加入到系统环境中）
import coco

  

# Directory to save logs and trained model
#保存日志和训练模型的文件夹
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
#训练权重文件的本地路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#如果需要的话，从Releases下载COCO训练权重
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#要运行检测的图像目录
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

#恭喜
#我们将使用在MS-COCO数据集上训练的模型。该模型的配置在coco.py的CocoConnfig类中
#对于推理来说，请稍微修改配置以适合任务。为此，请对CocoConfig类进行子类化并覆盖您需要更改的属性

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    #将batchsize设置成1因为我们将以此在一个图像上进行推断
    #Batch=GPU个数*每个GPU处理的图像数
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

#保存配置信息
config = InferenceConfig()
#输出配置信息在面板
#config.display()

# Create model object in inference mode.
#创建模型Model，调用了model.py中的Mask RCNN类
#对应参数（模型类型，模型文件夹，配置信息）
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
#载入模型权重文件
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
#将整数id与类型名字对应（如模型预测的id=1，对应的是person）
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
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

# Load a random image from the images folder
#返回文件夹中所有图片名字
file_names = next(os.walk(IMAGE_DIR))[2]
#随机选择一张图片（skimage模块用于图像的读取，显示，转换等）
image = skimage.io.imread("./E.jpeg")

# Run detection
#进行检测
results = model.detect([image], verbose=1)

# Visualize results
#结果可视化，result[0]是预测的标签信息
#display_instances也是mask_rcnn中的自定义函数
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
