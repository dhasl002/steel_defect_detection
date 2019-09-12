import tensorflow as tf

def conv2d(tempX, tempW):
	return tf.nn.conv2d(tempX, tempW, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def model_simple(x_image):
    W_conv1 = weight_variable([3, 3, 3, 64])
    h_conv1 = conv2d(x_image, W_conv1)
    norm_1 = tf.contrib.layers.batch_norm(h_conv1)
    relu_1 = tf.nn.relu(norm_1)

    lastWt = weight_variable([1, 1, 64, 2])
    lastConv = conv2d(relu_1, lastWt)
    print(lastConv)
    final_conv = tf.reshape(lastConv, [-1, 2])
    print(final_conv)
    return final_conv


def model(x_image):
	with tf.name_scope("Layer1_analysis"):
		W_conv1 = weight_variable([3, 3, 3, 64])
		h_conv1 = conv2d(x_image, W_conv1)
		norm_1 = tf.contrib.layers.batch_norm(h_conv1)
		relu_1 = tf.nn.relu(norm_1)

		W_conv2 = weight_variable([3, 3, 64, 64])
		h_conv2 = conv2d(relu_1, W_conv2)
		norm_2 = tf.contrib.layers.batch_norm(h_conv2)
		relu_2 = tf.nn.relu(norm_2)
  
	pool1 = max_pool_2x2(relu_2)

	with tf.name_scope("Layer2_analysis"):
		W_conv3 = weight_variable([3, 3, 64, 128])
		h_conv3 = conv2d(pool1, W_conv3)
		norm_3 = tf.contrib.layers.batch_norm(h_conv3)
		relu_3 = tf.nn.relu(norm_3)
  
		W_conv4 = weight_variable([3, 3, 128, 128])
		h_conv4 = conv2d(relu_3, W_conv4)
		norm_4 = tf.contrib.layers.batch_norm(h_conv4)
		relu_4 = tf.nn.relu(norm_4)
  
	pool2 = max_pool_2x2(relu_4)

	with tf.name_scope("Layer3_analysis"):
		W_conv5 = weight_variable([3, 3, 128, 256])
		h_conv5 = conv2d(pool2, W_conv5)
		norm_5 = tf.contrib.layers.batch_norm(h_conv5)
		relu_5 = tf.nn.relu(norm_5)
  
		W_conv6 = weight_variable([3, 3, 256, 256])
		h_conv6 = conv2d(relu_5, W_conv6)
		norm_6 = tf.contrib.layers.batch_norm(h_conv6)
		relu_6 = tf.nn.relu(norm_6)
  
	pool3 = max_pool_2x2(relu_6)

	with tf.name_scope("Layer4_analysis"):
		W_conv7 = weight_variable([3, 3, 256, 512])
		h_conv7 = conv2d(pool3, W_conv7)
		norm_7 = tf.contrib.layers.batch_norm(h_conv7)
		relu_7 = tf.nn.relu(norm_7)
  
		W_conv8 = weight_variable([3, 3, 512, 512])
		h_conv8 = conv2d(relu_7, W_conv8)
		norm_8 = tf.contrib.layers.batch_norm(h_conv8)
		relu_8 = tf.nn.relu(norm_8)

	pool4 = max_pool_2x2(relu_8)

	with tf.name_scope("Layer5_analysis"):
		W_conv9 = weight_variable([3, 3, 512, 1024])
		h_conv9 = conv2d(pool4, W_conv9)
		norm_9 = tf.contrib.layers.batch_norm(h_conv9)
		relu_9 = tf.nn.relu(norm_9)
  
		W_conv10 = weight_variable([3, 3, 1024, 1024])
		h_conv10 = conv2d(relu_9, W_conv10)
		norm_10 = tf.contrib.layers.batch_norm(h_conv10)
		relu_10 = tf.nn.relu(norm_10)

	upSample1 = tf.layers.conv2d_transpose(relu_10, filters = 1024, kernel_size = (2,2), strides = (2,2))
  
	concat1 = tf.concat([relu_8, upSample1],3)

	with tf.name_scope("Layer4_synthesis"):
		W_conv11 = weight_variable([3, 3, 1536, 512])
		h_conv11 = conv2d(concat1, W_conv11)
		norm_11 = tf.contrib.layers.batch_norm(h_conv11)
		relu_11 = tf.nn.relu(norm_11)
  
		W_conv12 = weight_variable([3, 3, 512, 512])
		h_conv12 = conv2d(relu_11, W_conv12)
		norm_12 = tf.contrib.layers.batch_norm(h_conv12)
		relu_12 = tf.nn.relu(norm_12)
  
	upSample2 = tf.layers.conv2d_transpose(relu_12, filters = 512, kernel_size = (2,2), strides = (2,2))

	concat2 = tf.concat([relu_6, upSample2],3)

	with tf.name_scope("Layer3_synthesis"):
		W_conv13 = weight_variable([3, 3, 768, 256])
		h_conv13 = conv2d(concat2, W_conv13)
		norm_13 = tf.contrib.layers.batch_norm(h_conv13)
		relu_13 = tf.nn.relu(norm_13)
  
		W_conv14 = weight_variable([3, 3, 256, 256])
		h_conv14 = conv2d(relu_13, W_conv14)
		norm_14 = tf.contrib.layers.batch_norm(h_conv14)
		relu_14 = tf.nn.relu(norm_14)
  
	upSample3 = tf.layers.conv2d_transpose(relu_14, filters = 256, kernel_size = (2,2), strides = (2,2))
  
	concat3 = tf.concat([relu_4, upSample3],3)
  
	with tf.name_scope("Layer2_synthesis"):
		W_conv15 = weight_variable([3, 3, 384, 128])
		h_conv15 = conv2d(concat3, W_conv15)
		norm_15 = tf.contrib.layers.batch_norm(h_conv15)
		relu_15 = tf.nn.relu(norm_15)
  
		W_conv16 = weight_variable([3, 3, 128, 128])
		h_conv16 = conv2d(relu_15, W_conv16)
		norm_16 = tf.contrib.layers.batch_norm(h_conv16)
		relu_16 = tf.nn.relu(norm_16)
  
	upSample4 = tf.layers.conv2d_transpose(relu_16, filters = 128, kernel_size = (2,2), strides = (2,2))
  
	concat4 = tf.concat([relu_2, upSample4],3)

	with tf.name_scope("Layer1_synthesis"):
		W_conv17 = weight_variable([3, 3, 192, 64])
		h_conv17 = conv2d(concat4, W_conv17)
		norm_17 = tf.contrib.layers.batch_norm(h_conv17)
		relu_17 = tf.nn.relu(norm_17)
  
		W_conv18 = weight_variable([3, 3, 64, 64])
		h_conv18 = conv2d(relu_17, W_conv18)
		norm_18 = tf.contrib.layers.batch_norm(h_conv18)
		relu_18 = tf.nn.relu(norm_18)

	lastWt = weight_variable([1, 1, 64, 2])
	lastConv = conv2d(relu_18, lastWt)
  
	final_conv = tf.reshape(lastConv, [-1, 2])
	print(final_conv)
	return final_conv
