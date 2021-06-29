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

#### Requeriments:
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
 <img src="./Screenshots/Train.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


<figure>
 <img src="./Screenshots/Test.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

<figure>
 <img src="./Screenshots/Valid.png" width="1072" alt="Combined Image" />
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

As variáveis `keep_prob` e `keep_prob_conv` são  utilizadas para controlar a taxa de dropout ao treinar a ConvNet.




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

Usando o VGGNet, obteve-se uma máxima precisão de **validação de 99,86% **. 

Este modelo será utilizado para predizer os labels do conjunto de teste.



---

## Step 5: Testing the Model using the Test Set

Utilizando o conjunto de testes, busca-se para medir a precisão do modelo em relação a exemplos desconhecidos
**Acurácia do teste = 97.2%**.


Em seguida, representa-se graficamente a matriz de confusão para ver onde o modelo realmente falha.

<figure>
 <img src="./traffic-signs-data/Screenshots/cm.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

A partir da matriz de confusão é possível observar que os vários limites de velocidade às vezes são classificados de forma errada entre si. Da mesma forma, os sinais de trânsito com formato traingular são classificados erroneamente entre si.

---

## Step 6: Testing the Model on New Images

Nesta etapa, utiliza-se o modelo para prever a classe dos sinais de trânsito de imagens aleatórias.
Número de novas imagens para teste:  5

<figure>
 <img src="./traffic-signs-data/Screenshots/NewImg.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

Essas imagens de teste incluem alguns sinais fáceis de prever e outros sinais são considerados difíceis de prever pelo modelo.


<figure>
 <img src="./traffic-signs-data/Screenshots/TopSoft.png" width="1072" alt="Combined Image" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Nota-se que nas 5 principais probabilidades de softmax, o modelo tem uma confiança muito alta quando se refere a prever sinais simples, como o sinal de "Pare" e "Sem entrada", e até mesmo alta confiança ao prever sinais triangulares simples sinais em uma imagem muito clara, como o sinal "Rendimento".

Por outro lado, a confiança do modelo diminui um pouco com o sinal triangular mais complexo em uma imagem "muito barulhenta", na imagem da placa "Pedestre", temos um sinal triangular com uma forma dentro dele, e os direitos autorais das imagens adicionam algum ruído. pela imagem, o modelo foi capaz de prever a verdadeira aula, mas com 80% de confiança.

E no sinal de "Limite de velocidade", podemos observar que o modelo previu com precisão que se trata de um sinal de "Limite de velocidade", mas foi de alguma forma confundido entre os diferentes limites de velocidade. No entanto, previu a verdadeira placa no final.

O modelo VGGNet foi capaz de prever a classe certa para cada uma das 5 novas imagens de teste.


Usando VGGNet, conseguimos atingir uma taxa de precisão muito alta. Podemos observar que os modelos saturam após quase 10 épocas, então podemos economizar alguns recursos computacionais e reduzir o número de épocas para 20. Também podemos tentar outras técnicas de pré-processamento para melhorar ainda mais a precisão do modelo. Podemos melhorar ainda mais o modelo usando CNNs hierárquicos para primeiro identificar grupos mais amplos (como sinais de trânsito) e, em seguida, ter CNNs para classificar características mais refinadas (como o limite de velocidade real). Este modelo funcionará apenas em exemplos de entrada onde os sinais de trânsito estão centralizados no meio da imagem. Não tem a capacidade de detectar sinais nos cantos da imagem.


---

## Deployment

Ao realizar o deployment de aplicações, em resumo cria-se versões dessas aplicações no correspondente no App Engine. De formar geral, pode-se implantar aplicativos inteiros, incluindo todo o código-fonte e arquivos de configuração, ou pode implantar e atualizar versões individuais ou arquivos de configuração. Existem diversas opções interessantes para o deploymente, destacando o Google Cloud.

Google Cloud Platform provides infrastructure as a service, platform as a service, and serverless computing environments.
