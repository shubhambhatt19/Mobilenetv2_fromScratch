import tensorflow as tf
from tensorflow.keras import Model
import config as cfg


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def bottleneck(inputs,filters,kernel_size,t, alpha,strides, skip = False):
	x = tf.keras.layers.Conv2D(filters,kernel_size=(1,1),strides=(1,1), padding= 'same')(inputs)
	x = tf.keras.layers.BatchNormalization(axis = -1)(x)
	x = tf.keras.layers.ReLU(6)(x)

	x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(strides,strides), padding= 'same')(x)
	x = tf.keras.layers.BatchNormalization(axis = -1)(x)
	x = tf.keras.layers.ReLU(6)(x)

	x = tf.keras.layers.Conv2D(filters*alpha,kernel_size=(1,1), strides=(1,1), padding= 'same')(x)
	x = tf.keras.layers.BatchNormalization(axis = -1)(x)

	if skip:
		x = tf.keras.layers.Add()([inputs, x])
	return x

def inverted_block(inputs,filters,kernel_size,t, alpha,strides, skip = False, n = 1):

	x = bottleneck(inputs,filters,kernel_size,t, alpha, strides=strides)

	for i in range(1,n):
		x = bottleneck(x,filters,kernel_size,t,  alpha,strides=1, skip = True)
	return x

def mobilenetv2(inputs,noOfClass, alpha = 1):

	first_filters = make_divisible(32 * alpha, 8)

	x = tf.keras.layers.Conv2D(first_filters, (3, 3))(inputs)

	x = inverted_block(x, filters=16,kernel_size=(3, 3),t=1, alpha=alpha,strides = 1,n = 1)
	x = inverted_block(x, filters=24,kernel_size=(3, 3),t=6, alpha=alpha,strides = 2,n = 2)
	x = inverted_block(x, filters=32,kernel_size=(3, 3),t=6, alpha=alpha,strides = 2,n = 3)
	x = inverted_block(x, filters=64,kernel_size=(3, 3),t=6, alpha=alpha,strides = 2,n = 4)
	x = inverted_block(x, filters=96,kernel_size=(3, 3),t=6, alpha=alpha,strides = 1,n = 3)
	x = inverted_block(x, filters=160,kernel_size=(3, 3),t=6, alpha=alpha,strides = 2,n = 3)
	x = inverted_block(x, filters=320,kernel_size=(3, 3),t=6, alpha=alpha,strides = 1,n = 1)

	if alpha > 1.0:
		last_filters = make_divisible(1280 * alpha, 8)
	else:
		last_filters = 1280

	x = tf.keras.layers.Conv2D(last_filters,kernel_size=1, strides = 1)(x)
	x = tf.keras.layers.BatchNormalization(axis = -1)(x)
	x = tf.keras.layers.ReLU(6)(x)

	x = tf.keras.layers.GlobalAveragePooling2D()(x)

	x = tf.keras.layers.Reshape((1, 1, last_filters))(x)

	x = tf.keras.layers.Conv2D(noOfClass, (1, 1),padding='same')(x)
	x = tf.keras.layers.Activation('sigmoid')(x)
	output = tf.keras.layers.Reshape((noOfClass,))(x)
	
	model = Model(inputs = inputs, outputs = output)
	return model




