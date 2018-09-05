from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import numpy as np

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import cv2

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.constant(1., dtype=tf.float32)
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
trainingset_mean = tf.constant([62.42,62.42,62.42], dtype=tf.float32)


img_str = tf.read_file('3128.jpg')
print(type(img_str))
img_decoded = tf.image.decode_png(img_str, channels=3)
print(type(img_decoded))
img_resized = tf.image.resize_images(img_decoded, [227, 227])
img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
# RGB -> BGR
img_bgr = img_centered[:, :, ::-1]
img = tf.reshape(img_bgr,(1,227,227,3))

# Initialize model
model = AlexNet(x, keep_prob, 2, [])

# Link variable to model output
score = model.fc8
softmax = tf.nn.softmax(score)
label = tf.placeholder(tf.float32,None,name='label')
prediction_list = tf.placeholder(tf.float32,None,name='prediction_list')
accuracy = tf.metrics.accuracy(label,prediction_list)
auc = tf.metrics.auc(label,prediction_list)

# create saver instance
saver = tf.train.Saver()

with tf.Session() as sess:    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, './finetune_alexnet/checkpoints/model_epoch3.ckpt')
         

    pred = sess.run(softmax, feed_dict={x: sess.run(img)})
    print(pred)
    predicted_label = pred.argmax(axis=1)
    print(predicted_label)
