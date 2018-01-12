#!/usr/bin/python
import picamera
import picamera.array
import time
import cv2
import numpy as np
import threading
import collections
import RPi.GPIO as GPIO
import tensorflow as tf
import os
import glob
from math import *

tf.logging.set_verbosity(tf.logging.INFO)

num_classes = 8
image_width = 51
image_height = 45
channels = 3

def convolutional_neural_network(features, labels, mode):
	input_layer = tf.reshape(features, shape=[-1, image_width, image_height, channels])
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=512,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu
		)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu
		)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu
		)
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	conv4 = tf.layers.conv2d(
		inputs=pool3,
		filters=256,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu
		)
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

	flat = tf.reshape(pool4, [-1, pool4.shape[1] * pool4.shape[2] * pool4.shape[3]])
	dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.8, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout, units=num_classes)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	return tf.contrib.learn.ModelFnOps(mode=mode, predictions=predictions)


# https://github.com/marcsto/rl/blob/master/src/fast_predict.py
class FastPredict:
	def _createGenerator(self):
		while True:
			yield self.next_features

	def __init__(self, estimator):
		self.estimator = estimator
		self.first_run = True

	def predict(self, features):
		self.next_features = features
		if self.first_run:
			self.predictions = self.estimator.predict(x=self._createGenerator())
			self.first_run = False
		return next(self.predictions)

model = FastPredict(tf.contrib.learn.Estimator(model_fn=convolutional_neural_network, model_dir="tmp"))

image_deque = collections.deque(maxlen=3)
class_list = ()

class ImageHandler(picamera.array.PiRGBAnalysis):
	def __init__(self, camera):
		super(ImageHandler, self).__init__(camera)

	def analyze(self, image):
		global image_deque
		image_deque.append({'image':image, 'time':time.time()})

class CameraThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global camera
		try:
			t = threading.currentThread()
			while getattr(t, "do_run", True):
				camera.wait_recording(0)
		finally:
			camera.stop_recording()


camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.sensor_mode = 1
camera.exposure_mode = 'sports'
handler = ImageHandler(camera)
camera.start_recording(handler, 'bgr')
worker = CameraThread()
worker.start()


cache = {
	'warp_matrix': None,
	'shape': None
}

def process_image(image):
	global cache
	if cache['warp_matrix'] is None:
		src = np.array([[34/100.*image.shape[1], 36/100.*image.shape[0]],
						[47/100.*image.shape[1], 36/100.*image.shape[0]],
						[95/100.*image.shape[1], 85/100.*image.shape[0]],
						[0/100.*image.shape[1], 85/100.*image.shape[0]]], np.float32)

		width =  int((src[2][0] - src[3][0])/3)
		height = src[2][1] - src[1][1]

		dst = np.array([[0, 0],
						[width, 0],
						[width, height],
						[0, height]], np.float32)

		warp_matrix = cv2.getPerspectiveTransform(src, dst)
		warp_back_matrix = cv2.getPerspectiveTransform(dst, src)

		cache['shape'] = (width, height)
		cache['warp_matrix'] = warp_matrix

	im = cv2.warpPerspective(image, cache['warp_matrix'], cache['shape'], flags=cv2.INTER_LINEAR & cv2.WARP_FILL_OUTLIERS)
	sums = cv2.resize(im, (int(ceil(im.shape[1]/12.0)), int(ceil(im.shape[0]/12.0))), interpolation=cv2.INTER_AREA)
	return sums

class ImageProcessor(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global model
		global image_deque
		global class_list
		t = threading.currentThread()
		while getattr(t, "do_run", True):
			try:
				if len(image_deque) == 0:
					print('image_deque empty')
					time.sleep(1)
				else:
					t = time.time()
					obj = image_deque.pop()
					res = model.predict(np.float32([process_image(obj['image'])]))
					weighted_mean = 0
					for i in range(num_classes):
						weighted_mean += res['probabilities'][i]*i

					print('cnn ' + str(weighted_mean) + ' ' + str(time.time()-t))
					class_list.append({'time': obj['time'], 'class': weighted_mean})
			except Exception as e:
				print('ImageProcessor', e)
				pass

worker = ImageProcessor()
worker.start()

goal_class = 3.5
params = ((0, 4), (2, 5), (3, 6), (3.5, 8), (5, 4), (6, 2), (8, 1)) # (class, required_time in seconds)
keep_for = 1
last_change = time.time()

class CarController(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global class_list
		global goal_class
		global params
		global keep_for
		global last_change
		GPIO.setmode(GPIO.BCM)
		GPIO.setup(17, GPIO.OUT) # bottom relay
		GPIO.setup(27, GPIO.OUT) # top relay

		GPIO.output(27, True)
		GPIO.output(17, True)

		t = threading.currentThread()
		while getattr(t, "do_run", True):
			try:
				t = time.time()
				class_list = [x for x in class_list if x['time'] > t - keep_for]
				if len(class_list) > 0:
					avg_class = 0
					for x in class_list:
						avg_class += x['class']
					avg_class /= len(class_list)

					if abs(goal_class - avg_class) > .25:
						for i in range(1, len(params), 1):
							if avg_class < params[i][0]:
								required_time = params[i-1][1] + (avg_class - params[i-1][0]) * (params[i][1] - params[i-1][1]) / (params[i][0] - params[i-1][0])
								if required_time < t - last_change:
									last_change = t

									if goal_class > avg_class: # accelerate
										print('acc', required_time, avg_class)
										GPIO.output(27, False)
										time.sleep(.2)
										GPIO.output(27, True)

									else: # decelerate
										print('dcc', required_time, avg_class)
										GPIO.output(17, False)
										time.sleep(.2)
										GPIO.output(17, True)

								break

				time.sleep(.1)
			except Exception as e:
				print('CarController', e)
				pass

worker = CarController()
worker.start()
