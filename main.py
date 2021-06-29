# # -*- coding: utf-8 -*-
# """
# Created on Tue Jun 29 06:51:50 2021

# @author: jeyds
# """

import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import cv2
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle
import csv
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.metrics import confusion_matrix

# is it using the GPU?
print(tf.test.gpu_device_name())

# Show current TensorFlow version
tf.__version__


def new_decorator(func):
    def wrap_func():
        print("Code would be here, before executing the func")
        func()
        print("Code here will execute after the func()")
    return wrap_func

@new_decorator
def func_needs_decorator():
    print("This function is in need of a Decorator")


func_needs_decorator()


#Carregar arquivos de dados salvos

# train_data = "./data_signs/train.pickle"
# valid_data = "./data_signs/valid.pickle"
# test_data = "./data_signs/test.pickle"

# with open(train_data, mode='rb') as f:
#     train = pickle.load(f)
# with open(valid_data, mode='rb') as f:
#     valid = pickle.load(f)
# with open(test_data, mode='rb') as f:
#     test = pickle.load(f)


#Imagem Resize (32,32,3)

def resize_cv(img):
    return cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)


#Mapear os IDclasses em signs 

path_labels = "./Traffic Signs/Labels.csv"
signs = []

with open(path_labels, 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames,None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()



# Mapear
def mapping(treino, teste):


  sizes = []  #tamanhos originais (sizes)
  coords =[] #Bound Boxes
  caminhos = [] #Camimhos para imagens
  lab = [] #IDclasses
  indx = []
  for i in range(4500):
    indx.append(random.randint(0, 39000))
    

#Percorrer o .csv
  with open(treino, 'r') as csvfile:
      signnames = csv.reader(csvfile, delimiter=',')
      next(signnames,None)
      for row in signnames:
          sizes.append((int(row[0]),int(row[1])))
          coords.append((int(row[2]),int(row[3]),int(row[4]),int(row[5])))
          caminhos.append('./Traffic Signs/'+ (row[7]))
          lab.append(int(row[6]))

      sizes_valid = []
      coords_valid = []
      caminhos_valid = []
      lab_valid = []
      for j in range(len(indx)):
          sizes_valid.append(sizes[indx[j]])
          coords_valid.append(coords[indx[j]])
          caminhos_valid.append(caminhos[indx[j]])
          lab_valid.append(lab[indx[j]])
      
      x = sorted(indx, key=int,  reverse=True)
      w = 0
      for i in x:
        if w == 0:
          del(sizes[i])
          del(coords[i])
          del(caminhos[i])
          del(lab[i])
        else:
          del(sizes[i-1])
          del(coords[i-1])
          del(caminhos[i-1])
          del(lab[i-1])
        w += 1

         #   #  criar o array das imagens    
      img_train =[]
      for i in caminhos:
        imagem = cv2.imread(i)
        r = resize_cv(imagem)
        img_train.append(r) 
      
      img_valid = []
      for i in caminhos_valid:
        imagem = cv2.imread(i)
        r = resize_cv(imagem)
        img_valid.append(r)       

      img_train = np.asarray(img_train)     
      img_valid = np.asarray(img_valid)     

      csvfile.close()
     
 #Percorrer o .csv
  
  sizes_test = []
  coords_test = []
  caminhos_test = []
  lab_test = []

  with open(teste, 'r') as csvfile:
      signnames = csv.reader(csvfile, delimiter=',')
      next(signnames,None)
      for row in signnames:
          sizes_test.append((int(row[0]),int(row[1])))
          coords_test.append((int(row[2]),int(row[3]),int(row[4]),int(row[5])))
          caminhos_test.append('./Traffic Signs/'+ (row[7]))
          lab_test.append(int(row[6]))    
     
      img_test = []
      for i in caminhos_test:
        imagem = cv2.imread(i)
        r = resize_cv(imagem)
        img_test.append(r)   

      img_test = np.asarray(img_test)        

      csvfile.close()    
     

  train ={'features':img_train,'labels':lab,'sizes':sizes,'coords':coords}
  
  valid = {'features':img_valid,'labels':lab_valid,'sizes':sizes_valid,'coords':coords_valid}
  test ={'features':img_test,'labels':lab_test,'sizes':sizes_test,'coords':coords_test}
  
  return train, test, valid

# #Mapeamento dados (TRAIN, TEST, VALID)

treino = "./Traffic Signs/Train.csv"
teste = "./Traffic Signs/Test.csv"
train, test, valid  = mapping(treino,teste)

# #Salvar dados

with open('./data_signs/train.pickle', 'wb') as f:
    pickle.dump(train, f)
with open('./data_signs/test.pickle', 'wb') as f:
    pickle.dump(train, f)   
with open('./data_signs/valid.pickle', 'wb') as f:
    pickle.dump(train, f)


#Váriaveis do dataset

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Número do exemplo de treinamentos
n_train = X_train.shape[0]

# Número do exemplo de teste
n_test = X_test.shape[0]

# Número do exemplo de validção
n_validation = X_valid.shape[0]

# Imagem shape
image_shape = X_train[0].shape

#Número de classes no dataset
n_classes = len(np.unique(y_train))

print("Número do exemplo de treinamentos: ", n_train)
print("Número do exemplo de teste: ", n_test)
print("Número do exemplo de validçãos: ", n_validation)
print("Imagem shape =", image_shape)
print("Número de classes no dataset =", n_classes)



def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
    Exibe uma lista de imagens em uma única figura com matplotlib.

    """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i+1)
        indx = random.randint(0, len(dataset))
        #Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap = cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()

# Plotting sample examples
list_images(X_train, y_train, "Exemplo do treinamento")
list_images(X_test, y_test, "Exemplo do teste")
list_images(X_valid, y_valid, "Validation example")






#histograma da contagem de imagens em cada classe única.

def histogram_plot(dataset, label):
    """
    Histograma dos dados de entrada.

    """
    hist, bins = np.histogram(dataset, bins=n_classes)
    width = 0.75 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color = 'r', edgecolor='blue')
    plt.xlabel(label)
    plt.ylabel("Número de imagens")
    plt.show()
    
    
#Traçando histogramas da contagem de cada sinal

histogram_plot(y_train, "Exemplos do treinamento (IDclasses)")
histogram_plot(y_test, "Exemplos do teste (IDclasses)")
histogram_plot(y_valid, "Exemplos de validação (IDclasses)")


#Misturar os dados de treinamento 

X_train, y_train = shuffle(X_train, y_train)

#Conveter uma imagem colorida para uma escala cinza

def gray(image):
    """
    Converta as imagens para a escala de cinza.

    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Imagens em escala cinza
gray_images = list(map(gray, X_train))
list_images(gray_images, y_train, "Imagem Cinza", "gray")


#Equalizar histogramas

def local_histo_equalize(image):
    """
    Eequalização de histograma local a imagens em tons de cinza.
       
    """
    kernel = morp.disk(30)
    img_local = rank.equalize(image, selem=kernel)
    return img_local

#Imagens após a equalização do histograma local

equalized_images = list(map(local_histo_equalize, gray_images))
list_images(equalized_images, y_train, "Imagem equalizada", "gray")

#Normalizar imagens

def image_normalize(image):
    """
    Normaliza as imagens na escala [0, 1].
       
    """
    image = np.divide(image, 255)
    return image

# Imagens depois da normalização

n_training = X_train.shape
normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
for i, img in enumerate(equalized_images):
    normalized_images[i] = image_normalize(img)
list_images(normalized_images, y_train, "Imagem normalizada", "gray")
normalized_images = normalized_images[..., None]

#Pré-processamento das imagens

def preprocess(data):
    """
    Aplicando as etapas de pré-processamento aos dados.
   
    """
    gray_images = list(map(gray, data))
    equalized_images = list(map(local_histo_equalize, gray_images))
    n_training = data.shape
    normalized_images = np.zeros((n_training[0], n_training[1], n_training[2]))
    for i, img in enumerate(equalized_images):
        normalized_images[i] = image_normalize(img)
    normalized_images = normalized_images[..., None]
    return normalized_images


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

# Validation set preprocessing
X_valid_preprocessed = preprocess(X_valid)
one_hot_y_valid = tf.one_hot(y_valid, 43)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        normalized_images, y_train = shuffle(normalized_images, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = normalized_images[offset:end], y_train[offset:end]
            sess.run(VGGNet_Model.training_operation, 
            feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})

        validation_accuracy = VGGNet_Model.evaluate(X_valid_preprocessed, y_valid)
        print("EPOCH {} : Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
    VGGNet_Model.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")
    
    
# Pré-processamento do conjunto de teste
X_test_preprocessed = preprocess(X_test)



with tf.Session() as sess:
    VGGNet_Model.saver.restore(sess, os.path.join(DIR, "VGGNet"))
    y_pred = VGGNet_Model.y_predict(X_test_preprocessed)
    test_accuracy = sum(y_test == y_pred)/len(y_test)
    print("Acurácia do teste = {:.1f}%".format(test_accuracy*100))
                    
    
#Plotar a matriz de confusão

cm = confusion_matrix(y_test, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = np.log(.0001 + cm)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Log da matriz de confusão normalizada')
plt.ylabel('Label verdadeiro')
plt.xlabel('Label predito')
plt.show()


# Carregando e redimensionando novas imagens de teste

new_test_images = []
path = './image_test/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_test_images.append(img)
new_IDs = [13, 3, 14, 27, 17]
print("Número de novos exemplos: ", len(new_test_images))


#Exibindo as imagens

plt.figure(figsize=(15, 16))
for i in range(len(new_test_images)):
    plt.subplot(2, 5, i+1)
    plt.imshow(new_test_images[i])
    plt.xlabel(signs[new_IDs[i]])
    plt.ylabel("Nova imagem de teste")
    plt.xticks([])
    plt.yticks([])
plt.tight_layout(pad=0, h_pad=0, w_pad=0)
plt.show()

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

test_accuracy = 0
for i in enumerate(new_test_images_preprocessed):
    accu = new_IDs[i[0]] == np.asarray(y_pred[i[0]])[0]
    if accu == True:
        test_accuracy += 0.2
print("Acurárcia das novas imagens de teste = {:.1f}%".format(test_accuracy*100))

plt.figure(figsize=(15, 16))
new_test_images_len=len(new_test_images_preprocessed)
for i in range(new_test_images_len):
    plt.subplot(new_test_images_len, 2, 2*i+1)
    plt.imshow(new_test_images[i]) 
    plt.title(signs[y_pred[i][0]])
    plt.axis('off')
    plt.subplot(new_test_images_len, 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), y_prob[i, :])
    labels = [signs[j] for j in y_pred[i]]
    plt.yticks(np.arange(1, 6, 1), labels)
plt.show()