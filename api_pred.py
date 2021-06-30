# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 08:59:27 2021

@author: jeyds
"""


import os
import requests
import pickle
import numpy as np
import cv2
import csv
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
from flask import Flask, request, abort, jsonify, send_from_directory

# from VG import VGGnet
from pre_process import preprocess

app = Flask(__name__)

@app.route("/predicao")

def ok():
    
    train_data = "./data_signs/train.pickle"
    valid_data = "./data_signs/valid.pickle"
    test_data = "./data_signs/test.pickle"

    with open(train_data, mode='rb') as f:
        train = pickle.load(f)
    with open(valid_data, mode='rb') as f:
        valid = pickle.load(f)
    with open(test_data, mode='rb') as f:
        test = pickle.load(f)
        
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    
    #Número de classes no dataset
    n_classes = len(np.unique(y_train)) 
    
    # Mapear os IDclasses em signs 
    
    path_labels = "./Traffic Signs/Labels.csv"
    signs = []
    
    with open(path_labels, 'r') as csvfile:
        signnames = csv.reader(csvfile, delimiter=',')
        next(signnames,None)
        for row in signnames:
            signs.append(row[1])
        csvfile.close()
    
    
    
    
    #Modelo VGGnet 
    
    class VGGnet:  
    
        def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
            # Hiperparâmetros
    
            self.mu = mu
            self.sigma = sigma
    
            # Camada 1 (Convolutional): Input = 32x32x1. Output = 32x32x32
            self.conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = self.mu, stddev = self.sigma))
            self.conv1_b = tf.Variable(tf.zeros(32))
            self.conv1   = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b
    
            # Ativação ReLu 
            self.conv1 = tf.nn.relu(self.conv1)
    
            # Camada 2 (Convolutional): Input = 32x32x32. Output = 32x32x32
            self.conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = self.mu, stddev = self.sigma))
            self.conv2_b = tf.Variable(tf.zeros(32))
            self.conv2   = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b
    
            # Ativação ReLu 
            self.conv2 = tf.nn.relu(self.conv2)
    
            # Camada 3 (Pooling): Input = 32x32x32. Output = 16x16x32
            self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)
    
            # Camada  4 (Convolutional): Input = 16x16x32. Output = 16x16x64
            self.conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = self.mu, stddev = self.sigma))
            self.conv3_b = tf.Variable(tf.zeros(64))
            self.conv3   = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b
    
            # Ativação ReLu 
            self.conv3 = tf.nn.relu(self.conv3)
    
            # Camada 5 (Convolutional): Input = 16x16x64. Output = 16x16x64
            self.conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = self.mu, stddev = self.sigma))
            self.conv4_b = tf.Variable(tf.zeros(64))
            self.conv4   = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b
    
            # Ativação ReLu 
            self.conv4 = tf.nn.relu(self.conv4)
    
            # Camada 6 (Pooling): Input = 16x16x64. Output = 8x8x64.
            self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv) # dropout
    
            # Camada 7 (Convolutional): Input = 8x8x64. Output = 8x8x128.
            self.conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = self.mu, stddev = self.sigma))
            self.conv5_b = tf.Variable(tf.zeros(128))
            self.conv5   = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b
    
            # Ativação ReLu 
            self.conv5 = tf.nn.relu(self.conv5)
    
            # Layer 8 (Convolutional): Input = 8x8x128. Output = 8x8x128.
            self.conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = self.mu, stddev = self.sigma))
            self.conv6_b = tf.Variable(tf.zeros(128))
            self.conv6   = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b
    
            # Ativação ReLu 
            self.conv6 = tf.nn.relu(self.conv6)
    
            # Camada 9 (Pooling): Input = 8x8x128. Output = 4x4x128.
            self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv) # dropout
    
            # Flatten. Input = 4x4x128. Output = 2048
            self.fc0   = flatten(self.conv6)
    
            # Layer 10 (Fully Connected): Input = 2048. Output = 128.
            self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = self.mu, stddev = self.sigma))
            self.fc1_b = tf.Variable(tf.zeros(128))
            self.fc1   = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b
    
            # Ativação ReLu 
            self.fc1    = tf.nn.relu(self.fc1)
            self.fc1    = tf.nn.dropout(self.fc1, keep_prob) # dropout
    
            # Camada 11 (Fully Connected): Input = 128. Output = 128
            self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = self.mu, stddev = self.sigma))
            self.fc2_b  = tf.Variable(tf.zeros(128))
            self.fc2    = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b
    
            # Ativação ReLu 
            self.fc2    = tf.nn.relu(self.fc2)
            self.fc2    = tf.nn.dropout(self.fc2, keep_prob) # dropout
    
            # Camada 12 (Fully Connected): Input = 128. Output = n_out
            self.fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean = self.mu, stddev = self.sigma))
            self.fc3_b  = tf.Variable(tf.zeros(n_out))
            self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b
    
            # Treinamento
    
            self.one_hot_y = tf.one_hot(y, n_out)
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
            self.loss_operation = tf.reduce_mean(self.cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
            self.training_operation = self.optimizer.minimize(self.loss_operation)
    
            # Acurácia
    
            self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
            self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    
            # Salvar variáveis
            self.saver = tf.train.Saver()
            
        def y_predict(self, X_data, BATCH_SIZE=64):
            num_examples = len(X_data)
            y_pred = np.zeros(num_examples, dtype=np.int32)
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x = X_data[offset:offset+BATCH_SIZE]
                y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                                    feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
            return y_pred
        
        def evaluate(self, X_data, y_data, BATCH_SIZE=64):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
                accuracy = sess.run(self.accuracy_operation, 
                                    feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples    

    # x - espaço reservado para um batch das inputs da imagem
    # y - espaço reservado para um batch dos labels da saída da imagem
    
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    
    keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
    keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers
    
    # Validation set preprocessing
    X_valid_preprocessed = preprocess(X_valid)
    
    #Parâmetros específicos para o treinamento
    EPOCHS = 30
    BATCH_SIZE = 64
    DIR = 'Saved_Models'
    
    #Pipiline do treinamento do modelo
    
    VGGNet_Model = VGGnet(n_out = n_classes)
    model_name = "VGGNet"



    # Pré-processamento do conjunto de teste
    X_test_preprocessed = preprocess(X_test)
    
    
    
    
    
    # with tf.Session() as sess:
    #     VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
    #     y_pred = VGGNet_Model.y_predict(X_test_preprocessed)
    #     test_accuracy = sum(y_test == y_pred)/len(y_test)
    #     print("Acurácia do teste = {:.1f}%".format(test_accuracy*100))
    
    
    
    
    
    
    
    
    new_test_images = []
    path = './api_image_test/'
    for image in os.listdir(path):
        img = cv2.imread(path + image)
        img = cv2.resize(img, (32,32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        new_test_images.append(img)
    # new_IDs = [13, 3, 14, 27, 17]
    print("Número de novos exemplos: ", len(new_test_images))
    
    #Pré-processamento dos novos dados de teste
    new_test_images_preprocessed = preprocess(np.asarray(new_test_images))
    
    #Predição do modelo treinado

    def y_predict_model(Input_data, top_k=5):
        """
        Gera as previsões do modelo sobre os dados de entrada e gera as principais probabilidades softmax.
           
        """
        num_examples = len(Input_data)
        y_pred = np.zeros((num_examples, top_k), dtype=np.int32)
        y_prob = np.zeros((num_examples, top_k))
        with tf.Session() as sess:
            VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
            y_prob, y_pred = sess.run(tf.nn.top_k(tf.nn.softmax(VGGNet_Model.logits), k=top_k), 
                                  feed_dict={x:Input_data, keep_prob:1, keep_prob_conv:1})
        return y_prob, y_pred
    
    y_prob, y_pred = y_predict_model(new_test_images_preprocessed)
    
    # test_accuracy = 0
    # for i in enumerate(new_test_images_preprocessed):
    #     accu = new_IDs[i[0]] == np.asarray(y_pred[i[0]])[0]
    #     if accu == True:
    #         test_accuracy += 0.2
    # print("Acurárcia das novas imagens de teste = {:.1f}%".format(test_accuracy*100))
    
    plt.figure(figsize=(15, 16))
    new_test_images_len=len(new_test_images_preprocessed)
    # for i in range(new_test_images_len):
    #     plt.subplot(new_test_images_len, 2, 2*i+1)
    #     plt.imshow(new_test_images[i]) 
    #     plt.title(signs[y_pred[i][0]])
    #     plt.axis('off')
    #     plt.subplot(new_test_images_len, 2, 2*i+2)
    #     plt.barh(np.arange(1, 6, 1), y_prob[i, :])
    #     labels = [signs[j] for j in y_pred[i]]
    #     plt.yticks(np.arange(1, 6, 1), labels)
    # plt.show()
    
    z = []
    h =[]
    for i in range(new_test_images_len):
      z.append(signs[y_pred[i][0]])
      h.append(max(y_prob[i, :])*100)
      
    UPLOAD_DIRECTORY = './api_image_test/'

    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
      path = os.path.join(UPLOAD_DIRECTORY, filename)
      if os.path.isfile(path):
        files.append(filename)
        
    d = []
    
    for i in range(new_test_images_len):
        d.append('y_pred('+files[i]+'):' + str(z[i]) + '  ' + str(np.trunc(h[i]))+'%' + '---' )

    a = ''
    for i in range(new_test_images_len):
      a = a + d[i]

    return a

UPLOAD_DIRECTORY = './api_image_test/'

@app.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)

@app.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)

@app.route("/files/<filename>", methods=["POST"])
def post_file(filename):
    """Upload a file."""

    if "/" in filename:
        # Return 400 BAD REQUEST
        abort(400, "no subdirectories allowed")

    with open(os.path.join(UPLOAD_DIRECTORY, filename), "wb") as fp:
        fp.write(request.data)

    # Return 201 CREATED
    return "", 201

app.run(host = "0.0.0.0", port = 2000, debug = False)







