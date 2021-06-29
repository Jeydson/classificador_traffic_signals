# classificador_traffic_signals
 Desafio Técnico FORD - Jeydson Lopes
 
 # Desafio Técnicno IEL/FORD - Reconhecer e classificar placas de trânsito
  Jeydson Lopes da Silva
  
**O projeto utiliza Python e TensorFlow para reconhecer e classificar placas de trânsito.**

**Dataset: [Traffic Signs / Kaggle Dataset](https://www.kaggle.com/venkateshroshan/traffic-signs).
**O dataset é composto por mais de 50.000 imagens distribuídas em 43 classes.**




## Pipeline:
- **Load Data**
- **Dataset Summary / Exploration / Featuring**
    - Shuffling
    - Grayscaling
    - Local Histogram Equalization
    - Normalization
- **Model**
    - VGGNet
- **Model Training and Evaluation**
- **Testing the Model Using the Test Set**
- **Testing the Model on New Images**
- **Deployment**



#### Environement:
-  Python 3.6.2
-  TensorFlow 1.15.0 (GPU support)

#### Requerimentst:
- requeriments.txt

---
## Step 1: Load Data


Os arquivos de dados para o modelo podem ser carregados previamente a partira da pasatas "data signs" ou  obtidos a partir 
de processamento dos arquivos .csv da pasta "Traffic signs".

Existem 3 arquvios `.pickle` com imagens previamente redimensionadas para o tamanho de 32x32:
- `train.pickle`: The training set.
- `test.pickle`: The testing set.
- `valid.pickle`: The validation set.

O processo para a obtenção desses arquivos é feito na etapa 2.

---

## Step 2: Dataset Summary / Exploration / Featuring

Os arquivos/dados são caracterizados por dicionários da seguinte forma:

- `'features'` é um array multidimensional (4D) dos pixels das imagens dos traffic signs (num examples, width, height, channels).
- `'labels'` é um array (1D) contendo o label/class do traffic sign.
- `'sizes'` é uma lista contendo tuplas (width, height), representando os a largura e tamanho orginais das imagens.
- `'coords'` é uma lista contendo tuplas, (x1, y1, x2, y2), representando as coordenadas dos bounding box em torno do sinal na imagem.

**Nesta aplicação, optou-se por utilizar as seguintes estruturas para os dados do dataset.**
Number of training examples:  34709
Number of testing examples:  12630
Number of validation examples:  4500
Image data shape = (32, 32, 3)
Number of classes = 43

**Assim, utilizou-se o `matplotlib` para exibir amostras de imagem de cada subconjunto.**


<figure>
 <img src="./traffic-signs-data/Screenshots/Train.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


<figure>
 <img src="./traffic-signs-data/Screenshots/Test.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

<figure>
 <img src="./traffic-signs-data/Screenshots/Valid.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


**Utilizando o `numpy` é exibido o histograma da contagem de imagens em cada classe única.**


<figure>
 <img src="./traffic-signs-data/Screenshots/TrainHist.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

<figure>
 <img src="./traffic-signs-data/Screenshots/TestHist.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

<figure>
 <img src="./traffic-signs-data/Screenshots/ValidHist.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

---

## Step 3: Dataset Summary / Exploration / Featuring

No passo 3 são aplicadas várias etapas de pré-processamento às imagens de entrada, obtendo assim melhores condições para o treinamento do modelo.

**Técnicas utilizadas:**

Shuffling.
Grayscaling.
Local Histogram Equalization.
Normalization.




1. **Shuffling**: De forma geral, mistura-se os dados de treinamento para aumentar a aleatoriedade e a variedade no conjunto de dados* de treinamento, para que o modelo seja mais estável. Usaremos sklearn para embaralhar nossos dados.

2. **Grayscaling**: Imagens em tons de cinza (ao invés de cores) melhora a precisão da rede neural. Nesse caso, utiliza-se o OpenCV para converter as imagens de treinamento em escala de cinza.

<figure>
 <img src="./traffic-signs-data/Screenshots/Gray.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

3. **Local Histogram Equalization**: Uma vez que o dataset utiliza de imagens reais (baixo contraste), utiliza-se esta técnica para distribuir os valores de intensidade mais frequentes em uma imagem, aprimorando assim essas imagens. Assim sendo, utiliza-se o skimage para aplicar a equalização do histograma local às imagens de treinamento.

<figure>
 <img src="./traffic-signs-data/Screenshots/Equalized.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

4. **Normalization**: Este é um processamento importante, pois os dados da imagem devem ser normalizados para que os dados tenham média zero e variância igual. A normalização busca modificar a faixa de valores da intensidade do pixel.

<figure>
 <img src="./traffic-signs-data/Screenshots/Normalized.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

---

## Step 3: Model

Nesta etapa, iremos projetar e implementar um modelo de aprendizado profundo que aprende a reconhecer sinais de trânsito do dataset Traffic Sign do Kaggle.

No modelo será utilizado redes convolucionais (ConvNets) para classificar as imagens deste conjunto de dados. De forma geral, as ConvNets são ideais para reconhecer padrões visuais diretamente dos pixels de imagens com pré-processamento mínimo. As ConvNets aprendem automaticamente hierarquias de recursos invariáveis em todos os níveis a partir dos dados.

O modelo é implementado utilizando o framework TensorFlow.

### 2.  VGGNet

A VGGNet apresenta uma melhoria em relação a algumas ConvNEts, como a Lenet por exemplo. Este fato se dá pela avaliação completa de redes de profundidade crescente usando uma arquitetura com filtros de convolução muito pequenos (3x3).

Nesse caso, a arquitetura VGGNet utilizará 12 camadas.

Input -- Convolution -- ReLU --Convolution -- ReLU --Pooling -- Convolution --ReLU -- Convolution -- ReLU -- Pooling -- Convolution -- ReLU -- Convolution -- ReLU -- Pooling -- FullyConnected -- ReLU -- FullyConnected -- ReLU -- FullyConnected

De formar a facilitar o desenvolvimento do modelo, a arquitetura, parâmetros e hiperâmetros

utilizados foram obtidos a partir de aplicações semelhantes de trabalhos relacionados.

**VGGNet architecture:**
<figure>
 <img src="VGGNet.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


**Layer 1 (Convolutional):** Saída shape = 32x32x32.

**Activation.** Relu.

**Layer 2 (Convolutional):** Saída shape = 2x32x32.

**Activation.** Relu.

**Layer 3 (Pooling)** Saída shape = 16x16x32.

**Layer 4 (Convolutional):** Saída shape = 16x16x64.

**Activation.** Relu.

**Layer 5 (Convolutional):** Saída shape = 16x16x64.

**Activation.** Relu.

**Layer 6 (Pooling)** Saída shape = 8x8x64.

**Layer 7 (Convolutional):** Saída shape = 8x8x128.

**Activation.** Relu.

**Layer 8 (Convolutional):** Saída shape = 8x8x128.

**Activation.** Relu.

**Layer 9 (Pooling)** Saída shape = 4x4x128.

**Flattening:** Saída 1D ao invés de 3D.

**Layer 10 (Fully Connected):** Saída = 128 outputs.

**Activation.** Relu.

**Layer 11 (Fully Connected):** Saída = 128 outputs.

**Activation.** Relu.

**Layer 12 (Fully Connected):** Saída = 43 outputs.

---

## Step 4: Model Training and Evaluation

Nesta etapa o modelo é utilizado para calcular a softmax cross entropy entre os logits e os labels das imagens.

De forma a minimizar este problema pode-se utilizaro Dropout. O Dropout é uma técnica busca descartar unidades aleatoriamente (junto com suas conexões) da rede neural durante o treinamento. Isso evita que as unidades se adaptem demais.

As variáveis `keep_prob` e `keep_prob_convsão utilizadas` para controlar a taxa de dropout ao treinar a ConvNet.




**Pipeline de treinamento para treinar o modelo**

- Antes de cada época, embaralharemos o conjunto de treinamento.
- Após cada época, mede-se a perda e a precisão do conjunto de validação.
- Após o treinamento, salva-se o modelo.
- Uma baixa precisão nos conjuntos de treinamento e validação implica em ajuste insuficiente. Uma alta precisão no conjunto de treinamento, mas baixa precisão no conjunto de validação implica overfitting.

### VGGNet Model
```
EPOCH 1 : Validation Accuracy = 35.089%
EPOCH 2 : Validation Accuracy = 69.711%
EPOCH 3 : Validation Accuracy = 84.378%
EPOCH 4 : Validation Accuracy = 91.600%
EPOCH 5 : Validation Accuracy = 96.200%
EPOCH 6 : Validation Accuracy = 97.600%
EPOCH 7 : Validation Accuracy = 97.822%
EPOCH 8 : Validation Accuracy = 97.756%
EPOCH 9 : Validation Accuracy = 98.800%
EPOCH 10 : Validation Accuracy = 98.911%
EPOCH 11 : Validation Accuracy = 99.422%
EPOCH 12 : Validation Accuracy = 99.644%
EPOCH 13 : Validation Accuracy = 99.600%
EPOCH 14 : Validation Accuracy = 99.578%
EPOCH 15 : Validation Accuracy = 99.800%
EPOCH 16 : Validation Accuracy = 99.711%
EPOCH 17 : Validation Accuracy = 99.822%
EPOCH 18 : Validation Accuracy = 99.733%
EPOCH 19 : Validation Accuracy = 99.711%
EPOCH 20 : Validation Accuracy = 99.800%
EPOCH 21 : Validation Accuracy = 99.778%
EPOCH 22 : Validation Accuracy = 99.822%
EPOCH 23 : Validation Accuracy = 99.800%
EPOCH 24 : Validation Accuracy = 99.644%
EPOCH 25 : Validation Accuracy = 99.844%
EPOCH 26 : Validation Accuracy = 99.867%
EPOCH 27 : Validation Accuracy = 99.622%
EPOCH 28 : Validation Accuracy = 99.867%
EPOCH 29 : Validation Accuracy = 99.867%
EPOCH 30 : Validation Accuracy = 99.756%
```

Using VGGNet, we've been able to reach a maximum **validation accuracy of 99.3%**. As you can observe, the model has nearly saturated after only 10 epochs, so we can reduce the epochs to 10 and save computational resources.

We'll use this model to predict the labels of the test set.


---

## Step 5: Testing the Model using the Test Set

Now, we'll use the testing set to measure the accuracy of the model over unknown examples.
We've been able to reach a **Test accuracy of 97.6%**. A remarkable performance.

Now we'll plot the confusion matrix to see where the model actually fails.

<figure>
 <img src="./traffic-signs-data/Screenshots/cm.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

We observe some clusters in the confusion matrix above. It turns out that the various speed limits are sometimes misclassified among themselves. Similarly, traffic signs with traingular shape are misclassified among themselves. We can further improve on the model using hierarchical CNNs to first identify broader groups (like speed signs) and then have CNNs to classify finer features (such as the actual speed limit).

---

## Step 6: Testing the Model on New Images

In this step, we will use the model to predict traffic signs type of 5 random images of German traffic signs from the web our model's performance on these images.
Number of new testing examples:  5

<figure>
 <img src="./traffic-signs-data/Screenshots/NewImg.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

These test images include some easy to predict signs, and other signs are considered hard for the model to predict.

For instance, we have easy to predict signs like the "Stop" and the "No entry". The two signs are clear and belong to classes where the model can predict with  high accuracy.

On the other hand, we have signs belong to classes where has poor accuracy, like the "Speed limit" sign, because as stated above it turns out that the various speed limits are sometimes misclassified among themselves, and the "Pedestrians" sign, because traffic signs with traingular shape are misclassified among themselves.

<figure>
 <img src="./traffic-signs-data/Screenshots/TopSoft.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

As we can notice from the top 5 softmax probabilities, the model has very high confidence (100%) when it comes to predict simple signs, like the "Stop" and the "No entry" sign, and even high confidence when predicting simple triangular signs in a very clear image, like the "Yield" sign.

On the other hand, the model's confidence slightly reduces with more complex triangular sign in a "pretty noisy" image, in the "Pedestrian" sign image, we have a triangular sign with a shape inside it, and the images copyrights adds some noise to the image, the model was able to predict the true class, but with 80% confidence.

And in the "Speed limit" sign, we can observe that the model accurately predicted that it's a "Speed limit" sign, but was somehow confused between the different speed limits. However, it predicted the true class at the end.

The VGGNet model was able to predict the right class for each of the 5 new test images. Test Accuracy = 100.0%.
In all cases, the model was very certain (80% - 100%).


---

## Conclusion

Using VGGNet, we've been able to reach a very high accuracy rate. We can observe that the models saturate after nearly 10 epochs, so we can save some computational resources and reduce the number of epochs to 10.
We can also try other preprocessing techniques to further improve the model's accuracy..
We can further improve on the model using hierarchical CNNs to first identify broader groups (like speed signs) and then have CNNs to classify finer features (such as the actual speed limit)
This model will only work on input examples where the traffic signs are centered in the middle of the image. It doesn't have the capability to detect signs in the image corners.
