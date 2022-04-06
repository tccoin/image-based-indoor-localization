# from scipy.misc import imread, imresize

# from keras.layers import Input, Dense, Conv2D
# from keras.layers import MaxPooling2D, AveragePooling2D
# from keras.layers import ZeroPadding2D, Dropout, Flatten
# from keras.layers import merge, Reshape, Activation, BatchNormalization, concatenate
# from keras.utils.conv_utils import convert_kernel
# from keras import backend as K
# from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, concatenate, Flatten, Dropout, BatchNormalization, Input
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda


beta = None


# def setBeta():
# 	global beta
# 	beta = run_siamese.TrainingHyperParams.beta

	
def euc_loss3x(y_true, y_pred):
	lx = keras.backend.sum(tf.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True)
	return 1 * lx


def euc_loss3q(y_true, y_pred):
	lq = keras.backend.sum(tf.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True)
	return beta * lq


def mseLossKeras(y_true, y_pred):
	global beta
	diff = tf.subtract(y_pred, y_true)
	diff2 = tf.square(diff)
	pos, ori = tf.split(diff2, [3, 4], 2)
	#ori *= beta
	mzsum = tf.reduce_sum(pos, axis=2) + beta * tf.reduce_sum(ori, axis=2)
	loss = tf.reduce_mean(mzsum, axis=None) # reduce over all dimensions
	return loss

#	Added by Jingwei
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network(input_shape):
	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
	#####################################################################
	# MITESH
	# https://www.tensorflow.org/tutorials/using_gpu
	# /gpu:0 should be used
	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
	#####################################################################
	# with tf.device('/gpu:0'):
	input = Input(shape=(224, 224, 3))

	conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1')(input)
	pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
	norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
	reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu', name='reduction2')(norm1)
	conv2 = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2')(reduction2)
	norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
	pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
	icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu', name='icp1_reduction1')(pool2)
	icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu', name='icp1_out1')(icp1_reduction1)
	icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu', name='icp1_reduction2')(pool2)
	icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='icp1_out2')(icp1_reduction2)
	icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
	icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp1_out3')(icp1_pool)
	icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp1_out0')(pool2)
	icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

	icp2_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp2_reduction1')(icp2_in)
	icp2_out1 = Conv2D(192, (3, 3), padding='same', activation='relu', name='icp2_out1')(icp2_reduction1)
	icp2_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp2_reduction2')(icp2_in)
	icp2_out2 = Conv2D(96, (5, 5), padding='same', activation='relu', name='icp2_out2')(icp2_reduction2)
	icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
	icp2_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp2_out3')(icp2_pool)
	icp2_out0 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp2_out0')(icp2_in)
	icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

	icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
	icp3_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu', name='icp3_reduction1')(icp3_in)
	icp3_out1 = Conv2D(208, (3, 3), padding='same', activation='relu', name='icp3_out1')(icp3_reduction1)
	icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu', name='icp3_reduction2')(icp3_in)
	icp3_out2 = Conv2D(48, (5, 5), padding='same', activation='relu', name='icp3_out2')(icp3_reduction2)
	icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
	icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp3_out3')(icp3_pool)
	icp3_out0 = Conv2D(192, (1, 1), padding='same', activation='relu', name='icp3_out0')(icp3_in)
	icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

	icp4_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu', name='icp4_reduction1')(icp3_out)
	icp4_out1 = Conv2D(224, (3, 3), padding='same', activation='relu', name='icp4_out1')(icp4_reduction1)
	icp4_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu', name='icp4_reduction2')(icp3_out)
	icp4_out2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp4_out2')(icp4_reduction2)
	icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)
	icp4_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp4_out3')(icp4_pool)
	icp4_out0 = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp4_out0')(icp3_out)
	icp4_out = concatenate(inputs=[icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

	icp5_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp5_reduction1')(icp4_out)
	icp5_out1 = Conv2D(256, (3, 3), padding='same', activation='relu', name='icp5_out1')(icp5_reduction1)
	icp5_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu', name='icp5_reduction2')(icp4_out)
	icp5_out2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp5_out2')(icp5_reduction2)
	icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)
	icp5_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp5_out3')(icp5_pool)
	icp5_out0 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp5_out0')(icp4_out)
	icp5_out = concatenate(inputs=[icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

	icp6_reduction1 = Conv2D(144, (1, 1), padding='same', activation='relu', name='icp6_reduction1')(icp5_out)
	icp6_out1 = Conv2D(288, (3, 3), padding='same', activation='relu', name='icp6_out1')(icp6_reduction1)
	icp6_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp6_reduction2')(icp5_out)
	icp6_out2 = Conv2D(64, (5, 5), padding='same', activation='relu', name='icp6_out2')(icp6_reduction2)
	icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)
	icp6_out3 = Conv2D(64, (1, 1), padding='same', activation='relu', name='icp6_out3')(icp6_pool)
	icp6_out0 = Conv2D(112, (1, 1), padding='same', activation='relu', name='icp6_out0')(icp5_out)
	icp6_out = concatenate(inputs=[icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

	icp7_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp7_reduction1')(icp6_out)
	icp7_out1 = Conv2D(320, (3, 3), padding='same', activation='relu', name='icp7_out1')(icp7_reduction1)
	icp7_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp7_reduction2')(icp6_out)
	icp7_out2 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp7_out2')(icp7_reduction2)
	icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)
	icp7_out3 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp7_out3')(icp7_pool)
	icp7_out0 = Conv2D(256, (1, 1), padding='same', activation='relu', name='icp7_out0')(icp6_out)
	icp7_out = concatenate(inputs=[icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

	icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)
	icp8_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu', name='icp8_reduction1')(icp8_in)
	icp8_out1 = Conv2D(320, (3, 3), padding='same', activation='relu', name='icp8_out1')(icp8_reduction1)
	icp8_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu', name='icp8_reduction2')(icp8_in)
	icp8_out2 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp8_out2')(icp8_reduction2)
	icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)
	icp8_out3 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp8_out3')(icp8_pool)
	icp8_out0 = Conv2D(256, (1, 1), padding='same', activation='relu', name='icp8_out0')(icp8_in)
	icp8_out = concatenate(inputs=[icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

	icp9_reduction1 = Conv2D(192, (1, 1), padding='same', activation='relu', name='icp9_reduction1')(icp8_out)
	icp9_out1 = Conv2D(384, (3, 3), padding='same', activation='relu', name='icp9_out1')(icp9_reduction1)
	icp9_reduction2 = Conv2D(48, (1, 1), padding='same', activation='relu', name='icp9_reduction2')(icp8_out)
	icp9_out2 = Conv2D(128, (5, 5), padding='same', activation='relu', name='icp9_out2')(icp9_reduction2)
	icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)
	icp9_out3 = Conv2D(128, (1, 1), padding='same', activation='relu', name='icp9_out3')(icp9_pool)
	icp9_out0 = Conv2D(384, (1, 1), padding='same', activation='relu', name='icp9_out0')(icp8_out)
	icp9_out = concatenate(inputs=[icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

	cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)
	cls3_fc1_flat = Flatten()(cls3_pool)
	cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
	dropout3 = Dropout(0.5)(cls3_fc1_pose)
	# 	cls3_fc_pose_all = Dense(7, name='cls3_fc_pose_all')(dropout3)
	descriptor = Dense(128, name='cls3_fc_pose_xyz')(dropout3)
	# cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(dropout3)
	#
	return keras.Model(inputs=input, outputs=[descriptor])

def create_siamese_keras(weights_path=None, tune=False):
	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
	#####################################################################
	# MITESH
	# https://www.tensorflow.org/tutorials/using_gpu 
	# /gpu:0 should be used
	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
	#####################################################################
	# with tf.device('/gpu:0'):
	input_a = Input(shape=(224, 224, 3))
	input_b = Input(shape=(224, 224, 3))

	# network definition
	base_network = create_base_network(input_shape=(224, 224, 3))

	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	siamesenet = keras.Model(inputs=[input_a, input_b], outputs=distance)
# 	posenet = keras.Model(inputs=input, outputs=[cls3_fc_pose_all])

	return siamesenet


def create_posenet_keras_2d(weights_path=None, tune=False):
	# creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
	#####################################################################
	# MITESH
	# https://www.tensorflow.org/tutorials/using_gpu 
	# /gpu:0 should be used
	# https://github.com/kentsommer/keras-posenet/issues/1 : as per this note it will pick up GPU is there is one
	#####################################################################
	# with tf.device('/gpu:0'):
	input = Input(shape=(224, 224, 3))

	conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu',  name='conv1')(input)
	pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
	norm1 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm1')(pool1)
	reduction2 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='reduction2')(norm1)
	conv2 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='conv2')(reduction2)
	norm2 = BatchNormalization(axis=3, epsilon=1e-6, momentum=0.9, name='norm2')(conv2)
	pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
	icp1_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp1_reduction1')(pool2)
	icp1_out1 = Conv2D(128, (3, 3), padding='same', activation='relu',   name='icp1_out1')(icp1_reduction1)
	icp1_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp1_reduction2')(pool2)
	icp1_out2 = Conv2D(32, (5, 5), padding='same', activation='relu',   name='icp1_out2')(icp1_reduction2)
	icp1_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_pool')(pool2)
	icp1_out3 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp1_out3')(icp1_pool)
	icp1_out0 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp1_out0')(pool2)
	icp2_in = concatenate(inputs=[icp1_out0, icp1_out1, icp1_out2, icp1_out3], axis=3, name='icp2_in')

	icp2_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_reduction1')(icp2_in)
	icp2_out1 = Conv2D(192, (3, 3), padding='same', activation='relu',   name='icp2_out1')(icp2_reduction1)
	icp2_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp2_reduction2')(icp2_in)
	icp2_out2 = Conv2D(96, (5, 5), padding='same', activation='relu',   name='icp2_out2')(icp2_reduction2)
	icp2_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_pool')(icp2_in)
	icp2_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp2_out3')(icp2_pool)
	icp2_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp2_out0')(icp2_in)
	icp2_out = concatenate(inputs=[icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

	icp3_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_in')(icp2_out)
	icp3_reduction1 = Conv2D(96, (1, 1), padding='same', activation='relu',   name='icp3_reduction1')(icp3_in)
	icp3_out1 = Conv2D(208, (3, 3), padding='same', activation='relu',   name='icp3_out1')(icp3_reduction1)
	icp3_reduction2 = Conv2D(16, (1, 1), padding='same', activation='relu',   name='icp3_reduction2')(icp3_in)
	icp3_out2 = Conv2D(48, (5, 5), padding='same', activation='relu',   name='icp3_out2')(icp3_reduction2)
	icp3_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_pool')(icp3_in)
	icp3_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp3_out3')(icp3_pool)
	icp3_out0 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp3_out0')(icp3_in)
	icp3_out = concatenate(inputs=[icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')

	icp4_reduction1 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp4_reduction1')(icp3_out)
	icp4_out1 = Conv2D(224, (3, 3), padding='same', activation='relu',   name='icp4_out1')(icp4_reduction1)
	icp4_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp4_reduction2')(icp3_out)
	icp4_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp4_out2')(icp4_reduction2)
	icp4_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp4_pool')(icp3_out)
	icp4_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp4_out3')(icp4_pool)
	icp4_out0 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp4_out0')(icp3_out)
	icp4_out = concatenate(inputs=[icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

	icp5_reduction1 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_reduction1')(icp4_out)
	icp5_out1 = Conv2D(256, (3, 3), padding='same', activation='relu',   name='icp5_out1')(icp5_reduction1)
	icp5_reduction2 = Conv2D(24, (1, 1), padding='same', activation='relu',   name='icp5_reduction2')(icp4_out)
	icp5_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp5_out2')(icp5_reduction2)
	icp5_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp5_pool')(icp4_out)
	icp5_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp5_out3')(icp5_pool)
	icp5_out0 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp5_out0')(icp4_out)
	icp5_out = concatenate(inputs=[icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

	icp6_reduction1 = Conv2D(144, (1, 1), padding='same', activation='relu',   name='icp6_reduction1')(icp5_out)
	icp6_out1 = Conv2D(288, (3, 3), padding='same', activation='relu',   name='icp6_out1')(icp6_reduction1)
	icp6_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp6_reduction2')(icp5_out)
	icp6_out2 = Conv2D(64, (5, 5), padding='same', activation='relu',   name='icp6_out2')(icp6_reduction2)
	icp6_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp6_pool')(icp5_out)
	icp6_out3 = Conv2D(64, (1, 1), padding='same', activation='relu',   name='icp6_out3')(icp6_pool)
	icp6_out0 = Conv2D(112, (1, 1), padding='same', activation='relu',   name='icp6_out0')(icp5_out)
	icp6_out = concatenate(inputs=[icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

	icp7_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp7_reduction1')(icp6_out)
	icp7_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp7_out1')(icp7_reduction1)
	icp7_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp7_reduction2')(icp6_out)
	icp7_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp7_out2')(icp7_reduction2)
	icp7_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp7_pool')(icp6_out)
	icp7_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp7_out3')(icp7_pool)
	icp7_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp7_out0')(icp6_out)
	icp7_out = concatenate(inputs=[icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

	icp8_in = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp8_in')(icp7_out)
	icp8_reduction1 = Conv2D(160, (1, 1), padding='same', activation='relu',   name='icp8_reduction1')(icp8_in)
	icp8_out1 = Conv2D(320, (3, 3), padding='same', activation='relu',   name='icp8_out1')(icp8_reduction1)
	icp8_reduction2 = Conv2D(32, (1, 1), padding='same', activation='relu',   name='icp8_reduction2')(icp8_in)
	icp8_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp8_out2')(icp8_reduction2)
	icp8_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp8_pool')(icp8_in)
	icp8_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp8_out3')(icp8_pool)
	icp8_out0 = Conv2D(256, (1, 1), padding='same', activation='relu',   name='icp8_out0')(icp8_in)
	icp8_out = concatenate(inputs=[icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

	icp9_reduction1 = Conv2D(192, (1, 1), padding='same', activation='relu',   name='icp9_reduction1')(icp8_out)
	icp9_out1 = Conv2D(384, (3, 3), padding='same', activation='relu',   name='icp9_out1')(icp9_reduction1)
	icp9_reduction2 = Conv2D(48, (1, 1), padding='same', activation='relu',   name='icp9_reduction2')(icp8_out)
	icp9_out2 = Conv2D(128, (5, 5), padding='same', activation='relu',   name='icp9_out2')(icp9_reduction2)
	icp9_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_pool')(icp8_out)
	icp9_out3 = Conv2D(128, (1, 1), padding='same', activation='relu',   name='icp9_out3')(icp9_pool)
	icp9_out0 = Conv2D(384, (1, 1), padding='same', activation='relu',   name='icp9_out0')(icp8_out)
	icp9_out = concatenate(inputs=[icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

	cls3_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid', name='cls3_pool')(icp9_out)
	cls3_fc1_flat = Flatten()(cls3_pool)
	cls3_fc1_pose = Dense(2048, activation='relu', name='cls3_fc1_pose')(cls3_fc1_flat)
	dropout3 = Dropout(0.5)(cls3_fc1_pose)
# 	cls3_fc_pose_all = Dense(7, name='cls3_fc_pose_all')(dropout3)
	cls3_fc_pose_xy = Dense(3, name='cls3_fc_pose_xy')(dropout3)
	cls3_fc_pose_yaw = Dense(4, name='cls3_fc_pose_yaw')(dropout3)
	#

	distance = Lambda(euclidean_distance,
					  output_shape=eucl_dist_output_shape)([processed_a, processed_b])
	posenet = keras.Model(inputs=input, outputs=[cls3_fc_pose_xy, cls3_fc_pose_yaw])
# 	posenet = keras.Model(inputs=input, outputs=[cls3_fc_pose_all])

	if tune:
		if weights_path:
			weights_data = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
			for layer in posenet.layers:
				if layer.name in weights_data.keys():
					layer_weights = weights_data[layer.name]
					layer.set_weights((layer_weights['weights'], layer_weights['biases']))
					#print("FINISHED SETTING THE WEIGHTS!")
	return posenet


if __name__ == "__main__":
	print("Please run either test.py or train.py to evaluate or fine-tune the network!")
