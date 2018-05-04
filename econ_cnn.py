from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import scipy.misc as sc
import os
import json
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib
from sklearn import metrics

class CNN(object):

  def __init__(self,days_past):
    """Model function for CNN."""
    self.input_features = tf.placeholder(dtype=tf.float32, name = "input_features")
    self.data_labels = tf.placeholder(dtype=tf.int64, name = "data_labels")
    self.onehot_labels = tf.one_hot(self.data_labels, depth=2, on_value=1, off_value=0)
    self.training = tf.placeholder(dtype=tf.bool, name = "training")
    self.learning_rate = tf.placeholder(dtype=tf.float32,name="learning_rate")

    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    self.days_lookback = days_past
    self.input_layer = tf.reshape(self.input_features, [-1, 8, 8, 1],name="input_layer")

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, days_lookback, 8, 1]
    # Output Tensor Shape: [batch_size, days_lookback, 8, 100]
    self.conv1 = tf.layers.conv2d(
        inputs=self.input_layer,
        filters=100,#64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv1")

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, days_lookback, 8, 100]
    # Output Tensor Shape: [batch_size,days_lookback/2, 4, 100]
    self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2,name="pool1")

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, days_lookback/2, 4, 100]
    # Output Tensor Shape: [batch_size, days_lookback/2, 4, 100]
    self.conv2 = tf.layers.conv2d(
        inputs=self.pool1,
        filters=100,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        name="conv2")

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, days_lookback/2, 4, 100]
    # Output Tensor Shape: [batch_size, days_lookback/4, 2, 100]
    self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2,name="pool2")

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, days_lookback/4, 2, 100]
    # Output Tensor Shape: [batch_size, days_lookback/4 * 2 * 100]
    self.pool2_flat = tf.reshape(self.pool2, [-1, int(self.days_lookback/4) * 2 * 100], name="cnn_features")

    # Combine CNN features with other input features.
    self.combined_inputs = tf.concat([self.pool2_flat],axis=0,name="combined_inputs")

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 16 * 16 * 256] # NOT UPDATED - combined inputs
    # Output Tensor Shape: [batch_size, 200]
    self.dense = tf.layers.dense(inputs=self.combined_inputs, units=200, activation=tf.nn.relu, name="dense")
    # Dense Layer
    # Densely connected layer with 512 neurons
    # Input Tensor Shape: [batch_size, 200] # NOT UPDATED - combined inputs
    # Output Tensor Shape: [batch_size, 100]
    self.dense2 = tf.layers.dense(inputs=self.dense,units=100,activation=tf.nn.relu,name="dense2")

    # Add dropout operation; 0.6 probability that element will be kept
    self.dropout = tf.layers.dropout(
        inputs=self.dense2, rate=0.4, training=self.training,name="dropout")

    # Logits layer
    # Input Tensor Shape: [batch_size, 512]
    # Output Tensor Shape: [batch_size, 2]. 2 for down and non-down.
    self.logits = tf.layers.dense(inputs=self.dropout, units=2,name="logits")
    self.probs = tf.nn.softmax(self.logits,name="probs")
    self.predicted_classes = tf.argmax(input=self.logits,axis=1,name="predicted_classes")
    self.accuracy = tf.contrib.metrics.accuracy(self.predicted_classes,self.data_labels)
    self.out_acc = tf.identity(self.accuracy,name="graph_accuracy")

    # Calculate Loss
    self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.onehot_labels, logits=self.logits)
    self.out_loss = tf.identity(self.loss,name="loss")
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step(),name = "train_op")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cnn_traintest(data_tuple,specs={"num_epochs":5,"days_past":8},verbose=False):
  SAVE_ON = ("save_dir" in specs)
  LOAD_ON = ("load_dir" in specs) and ("modelmeta_filepath" in specs)

  ######################## Load model and initialize saver
  days_past = specs["days_past"]
  if LOAD_ON:
    modelmeta_filepath = specs["modelmeta_filepath"]
    modelcheckpoint_dir = specs["load_dir"]
    sess = tf.Session()
    saver = tf.train.import_meta_graph(modelmeta_filepath)
    saver.restore(sess,tf.train.latest_checkpoint(modelcheckpoint_dir))
  else:
    econCNN = CNN(days_past)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

  if SAVE_ON:
    save_dir = specs["save_dir"]
    saver = tf.train.Saver()

  ######################## Grab data from specs
  num_epochs = specs["num_epochs"]
  train = data_tuple["train"]
  test = data_tuple["test"] #+ data_tuple["validation"]

  ######################## Split data into up and down sets
  test_up = []
  test_down = []
  for page in test:
    if page[1] == 0:
      test_up.append(page)
    else:
      test_down.append(page)
  train_up = []
  train_down = []
  for page in train:
    if page[1] == 0:
      train_up.append(page)
    else:
      train_down.append(page)
  train_upsampled = train_up+train_down

  ####################### Split sets into input features and labels
  x_test = np.asarray([np.asarray(x[0]) for x in test])
  y_test = np.asarray([x[1] for x in test])
  z_test = [x[2] for x in test]
  x_train = np.asarray([np.asarray(x[0]) for x in train])
  y_train = np.asarray([x[1] for x in train])
  z_train = [x[2] for x in train]
  x_train_up = np.asarray([np.asarray(x[0]) for x in train_up])
  y_train_up = np.asarray([0 for x in train_up])
  z_train_up = [x[2] for x in train_up]
  x_train_down = np.asarray([np.asarray(x[0]) for x in train_down])
  y_train_down = np.asarray([1 for x in train_down])
  z_train_down = [x[2] for x in train_down]
  x_test_up = np.asarray([np.asarray(x[0]) for x in test_up])
  y_test_up = np.asarray([0 for x in test_up])
  z_test_up = [x[2] for x in test_up]
  x_test_down = np.asarray([np.asarray(x[0]) for x in test_down])
  y_test_down = np.asarray([1 for x in test_down])
  z_test_down = [x[2] for x in test_down]
  train_labels = [x[1] for x in train]
  print("Num train: ",len(train))
  print("Num train up: ",len(train_up))
  print("Num train down: ",len(train_down))
  print("Num test: ",len(test))
  print("Num test up: ",len(x_test_up))
  print("Num test down: ",len(x_test_down))

  ####################### Batch everything
  BATCH_SIZE = 100
  x_trains = list(chunks(x_train,BATCH_SIZE))
  y_trains = list(chunks(y_train,BATCH_SIZE))
  x_tests = list(chunks(x_test,BATCH_SIZE))
  y_tests = list(chunks(y_test,BATCH_SIZE))
  z_tests = list(chunks(z_test,BATCH_SIZE))
  x_trains_up = list(chunks(x_train_up,BATCH_SIZE))
  y_trains_up = list(chunks(y_train_up,BATCH_SIZE))
  z_trains_up = list(chunks(z_train_up,BATCH_SIZE))
  x_trains_down = list(chunks(x_train_down,BATCH_SIZE))
  y_trains_down = list(chunks(y_train_down,BATCH_SIZE))
  z_trains_down = list(chunks(z_train_down,BATCH_SIZE))
  x_tests_up = list(chunks(x_test_up,BATCH_SIZE))
  y_tests_up = list(chunks(y_test_up,BATCH_SIZE))
  z_tests_up = list(chunks(z_test_up,BATCH_SIZE))
  x_tests_down = list(chunks(x_test_down,BATCH_SIZE))
  y_tests_down = list(chunks(y_test_down,BATCH_SIZE))
  z_tests_down = list(chunks(z_test_down,BATCH_SIZE))

  ####################### Training loop
  logs = {"NumTraindown" : len(train_down), 
          "NumTrainup" : len(train_up), 
          "NumTestdown" : len(x_test_down), 
          "NumTestup" : len(x_test_up)}
  epoch_logs = []
  train_count = len(train)
  print("STARTING CNN TRAINING")
  overfitting = False
  countdown = 20
  train_accs = []
  test_accs = []
  up_accs = []
  down_accs = []
  for i in range(num_epochs):
    shuffle(train_up)
    x_train_all = np.asarray([np.asarray(x[0]) for x in train_upsampled])
    y_train_all = np.asarray([x[1] for x in train_upsampled])
    for start, end in zip(range(0, train_count, BATCH_SIZE),range(BATCH_SIZE, train_count + 1,BATCH_SIZE)):
      sess.run("train_op",
        feed_dict={
          "input_features:0":x_train_all[start:end],
          "data_labels:0":y_train_all[start:end],
          "training:0":True,"learning_rate:0":0.001})

    # Track progress across epochs
    if True or (num_epochs > 20 and (i+1) % int(num_epochs/50) == 0) or (i == num_epochs-1):
      acc_results = []
      for x_list,y_list in [(x_trains,y_trains),
                            (x_tests,y_tests),
                            (x_tests_up,y_tests_up),
                            (x_tests_down,y_tests_down)]:
        totalCorrect = 0
        totalNum = 0
        for j in range(len(x_list)):
          x = x_list[j]
          y = y_list[j]
          acc = sess.run("graph_accuracy:0",
            feed_dict={
              "input_features:0":x,
              "data_labels:0":y,
              "training:0":False,})
          numCorrect = acc*len(x)
          totalCorrect += numCorrect
          totalNum += len(x)
        if totalNum == 0:
          print("Oops!")
          acc_results.append(0)
        else:
          acc_results.append(float(totalCorrect)/totalNum)

      acc_train = acc_results[0]
      acc_test = acc_results[1]
      acc_test_up = acc_results[2]
      acc_test_down = acc_results[3]
      train_accs.append(acc_train)
      test_accs.append(acc_test)
      up_accs.append(acc_test_up)
      down_accs.append(acc_test_down)


      if verbose:
        print("Epoch " + str(i))
        print("Training accuracy: " + str(acc_train))
        print("Testing accuracy: " + str(acc_test))
        print("Test up accuracy: " + str(acc_test_up))
        print("Test down accuracy: " + str(acc_test_down))
      
      epoch_logs.append([
        i,
        float(acc_train),
        float(acc_test),
        float(acc_test_up),
        float(acc_test_down),
        1-float(acc_test_up), 
        1-float(acc_test_down)])

      if SAVE_ON:
        savefile = save_dir+"cnn_model" + str(i)
        saver.save(sess,savefile)
        saver.export_meta_graph(savefile+".meta")
        
      if acc_train > 0.99 and not overfitting:
        overfitting = True
        print("Starting to overfit on training set. Ending in " + str(countdown) + " epochs.")
      elif acc_train > 0.99:
        countdown -= 1
        if countdown <= 0:
          print("Terminating epochs early due to overfit on training set.")
          break

  save_epoch_size = 50
  epoch_log_list = []
  for i in range(len(epoch_logs),save_epoch_size):
    epoch_log_list.append(epoch_logs[i:i+50])

  id_probs = []
  for x_list,y_list,z_list in [
    (x_trains_up,y_trains_up,z_trains_up),
    (x_trains_down,y_trains_down,z_trains_down),
    (x_tests_up,y_tests_up,z_tests_up),
    (x_tests_down,y_tests_down,z_tests_down),]:
    prob_tuples = []
    seen_ids = []
    for j in range(len(x_list)):
      x = x_list[j]
      y = y_list[j]
      z = z_list[j]
      if z in seen_ids:
        continue
      else:
        seen_ids.append(z)
      probs = sess.run("probs:0",
        feed_dict={
          "input_features:0":x,
          "data_labels:0":y,
          "training:0":False,})
      prob_tuple = [(z[i],int(y[i]),probs[i].tolist()) for i in range(len(probs))]
      prob_tuples.extend(prob_tuple)
    id_probs.append(prob_tuples)

  return_list = []#poch_log_list
  return_list.append(logs)
  return_list.append(id_probs)

  sess.close()
  tf.reset_default_graph()
  return id_probs, train_accs,test_accs,up_accs,down_accs
  