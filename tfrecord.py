import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from object_detection.utils import dataset_util

import io
from PIL import Image
writer = tf.io.TFRecordWriter("tfrecords/train.record")

BASE_PATH = "../cats-v-dogs_data/"
for dir_ in os.listdir(BASE_PATH):
    for sub_dir in os.listdir(BASE_PATH+dir_):
        for img_name in os.listdir(BASE_PATH+dir_+"/"+sub_dir):
            print(os.path.join(BASE_PATH+dir_+"/"+sub_dir))
            img = cv2.imread(os.path.join(BASE_PATH+dir_+"/"+sub_dir)+"/"+img_name)
            index = 0 if sub_dir == "cat" else 1
            class_name = sub_dir
            print(index, class_name)
            print(os.path.join(BASE_PATH+dir_+"/"+sub_dir)+"/"+img_name)
            with tf.io.gfile.GFile(os.path.join(BASE_PATH+dir_+"/"+sub_dir)+"/"+img_name, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            try:
                image = Image.open(encoded_jpg_io)
            except:
                continue
            if img is not None:
                # encoded_jpg = cv2.imencode('.jpg', img)[1].tostring()
                filename = tf.compat.as_bytes(img_name)
                image_format = b'jpg'

                classes_text = []
                classes = []
                width = img.shape[1]
                height = img.shape[0]

                classes_text.append(tf.compat.as_bytes(class_name))
                classes.append(index)

                tf_example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/source_id': dataset_util.bytes_feature(filename),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes)
                }))
                writer.write(tf_example.SerializeToString())
writer.close()
