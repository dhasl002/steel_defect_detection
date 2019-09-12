from os import listdir
from os.path import isfile, join
import cv2 
import time
import tensorflow as tf
import numpy as np
from model import model, model_simple
import cv2
import random
import csv

training_dir = "./train_images/"
batch_Size = 4 
width = 1600 
height = 256
training_images = [f for f in listdir(training_dir) if isfile(join(training_dir, f))]

def get_training_points():
    training_points = {}
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            training_points[row[0]] = row[2]
    print('Processed training data')
    return training_points

def convert_training_points(training_points):
    converted_training_points = {}
    all_points = []
    for image_path in training_images:
        encoded_points_str = training_points[image_path]
        encoded_points_arr = encoded_points_str.split(' ')
        for i in range(0, len(encoded_points_arr)-1, 2):
            location = int(encoded_points_arr[i])
            duration = int(encoded_points_arr[i+1])
            for i in range(0, duration-1):
                all_points.append(location + i)
        converted_training_points[image_path] = all_points
    print("converted training points")
    return converted_training_points

training_points = get_training_points()
converted_training_points = convert_training_points(training_points)

def train(batch_Size, width, height):
    a = np.zeros((batch_Size, height, width, 3),dtype = "float64")
    b = np.zeros((batch_Size, height * width, 2),dtype = "float64")
    for it in range(0, batch_Size):
        randImage = random.randint(0, 12568)
        rawImage = cv2.imread(training_images[randImage], 1)
        rawImageArray = np.asarray( rawImage, dtype="float64")
        groundTruthArray = np.zeros([height * width, 2])
        num = 0
        for point in converted_training_points[training_images[randImage]]:
            groundTruthArray[point][1] = 1
        for i in range(0, len(groundTruthArray)):
            if groundTruthArray[i][0] == 0 and groundTruthArray[i][1] == 0:
                groundTruthArray[i][0] = 1
                num += 1 
        
        a[it] = rawImageArray
        b[it] = groundTruthArray
        print(num)

    np.set_printoptions(threshold=np.inf)
    runTrainingBatch(a, b)
    a = np.zeros((batch_Size, height, width, 3),dtype = "float64")
    b = np.zeros((batch_Size, height*width, 2),dtype = "float64")

def runTrainingBatch(a, b):
    keep_prob = tf.placeholder(tf.float32)
    CurCross = cross_entropy.eval(feed_dict={x_image: a, y_: b})
    train_accuracy = accuracy.eval(feed_dict={x_image: a, y_: b, keep_prob: .5})
    train_step.run(feed_dict={x_image: a, y_: b, keep_prob: .5})
    print('training accuracy %g' % (train_accuracy))
    print('Loss %g' % (CurCross))

x_image = tf.placeholder(tf.float32, shape=[batch_Size, height, width, 3])
y_ = tf.placeholder(tf.float32, shape=[None, None, 2])

modelResult = model_simple(x_image) #sets modelResults to the final result of the model 

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=modelResult, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(modelResult, 1), tf.argmax(tf.reshape(y_, [-1, 2]), 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for curEpoch in range(1, 500):
		print('Epoch: ' + str(curEpoch) + '\n')
		train(batch_Size, width, height)
