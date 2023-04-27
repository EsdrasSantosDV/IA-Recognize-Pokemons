import numpy as np
import pandas as pd
import cv2 as cv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import requests

from PIL import Image
from io import BytesIO

from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

warnings.filterwarnings('ignore')

# Obtendo o diretório atual e definindo o diretório do conjunto de dados
current_directory = os.getcwd()
completodiretorio = current_directory + '/dataset'

# Carregando os nomes das classes e contando a quantidade de imagens em cada classe
classes = os.listdir(completodiretorio)

# Exibindo a quantidade de imagens em cada classe
contador = {}
for c in classes:
    contador[c] = len(os.listdir(os.path.join(completodiretorio, c)))

# Selecionando as 5 classes com mais imagens e armazenando-as na lista
classesselecionadas = sorted(contador.items(), key=lambda x: x[1], reverse=True)[:5]

classesselecionadas = [i[0] for i in classesselecionadas]

# Criando listas vazias para armazenar imagens e rótulos, e redimensionando as imagens para (96, 96):
X = []
Y = []

# Carregando e redimensionando imagens
for c in classes:
    if c in classesselecionadas:
        dir_path = os.path.join(completodiretorio, c)
        label = classesselecionadas.index(c)

        for i in os.listdir(dir_path):
            image = cv.imread(os.path.join(dir_path, i))

            try:
                resized = cv.resize(image, (96, 96))
                X.append(resized)
                Y.append(label)

            except:
                print(os.path.join(dir_path, i), '[Erro Não pode abrir o arquivo]')
                continue

obj = Counter(Y)

X = np.array(X).reshape(-1, 96, 96, 3)

# Converte as listas de imagens e rótulos em arrays NumPy e normalizando as imagens dividindo por 255.
X = X / 255.0

y = to_categorical(Y, num_classes=len(classesselecionadas))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=666)

#Criando um gerador de imagens para aumentar a quantidade de dados
datagen = ImageDataGenerator(rotation_range=45,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             shear_range=0.2)

datagen.fit(X_train)

#Arquitetura da CNN
model = Sequential()
model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(96, 96, 3), kernel_initializer='he_normal'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(256, 3, padding='same', kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(classesselecionadas), activation='softmax'))

checkpoint = ModelCheckpoint(current_directory + "/best_model.hdf5", verbose=1, monitor='val_accuracy',
                             save_best_only=True)

#Compilando o Modelo de Dados
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Treinando o Modelo de Dados
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=30,
                              validation_data=[X_test, y_test],
                              steps_per_epoch=len(X_train) // 32, callbacks=[checkpoint])


#Exibindo a evolução da acurácia e da perda durante o treinamento
fig = plt.figure(figsize=(17, 4))

plt.subplot(121)
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.grid()
plt.title(f'accuracy')
plt.show()

plt.subplot(122)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.title(f'loss')
plt.show()

model.load_weights(current_directory + "/best_model.hdf5")

model.save(current_directory + "/model.hdf5")


def carregar_imagem_local(caminho):
    imagem = cv.imread(caminho)
    return imagem


mewtwo = ['imagens/mewtwo1.jpg', 'imagens/mewtwo2.jpg', 'imagens/mewtwo3.jpg', 'imagens/mewtwo4.jpg',
          'imagens/mewtwo5.jpg']
pikachu = ['imagens/pikachu1.jpg', 'imagens/pikachu2.jpg', 'imagens/pikachu3.jpg', 'imagens/pikachu4.jpg',
           'imagens/pikachu5.jpg']
charmander = ['imagens/charmander1.jpg', 'imagens/charmander2.jpg', 'imagens/charmander3.jpg',
              'imagens/charmander4.jpg', 'imagens/charmander5.jpg']
bulbasaur = ['imagens/bulbasaur1.jpg', 'imagens/bulbasaur2.jpg', 'imagens/bulbasaur3.jpg', 'imagens/bulbasaur4.jpg',
             'imagens/bulbasaur5.jpg']
squirtle = ['imagens/squirtle1.jpg', 'imagens/squirtle2.jpg', 'imagens/squirtle3.jpg', 'imagens/squirtle4.jpg',
            'imagens/squirtle5.jpg']

test_df = [mewtwo, pikachu, charmander, bulbasaur, squirtle]

val_x = []
val_y = []

for i, urls in enumerate(test_df):
    for caminho in urls:
        imagem = carregar_imagem_local(caminho)
        val_x.append(imagem)
        val_y.append(i)

rows = 5
cols = 5

fig = plt.figure(figsize=(25, 25))

for i, j in enumerate(zip(val_x, val_y)):
    orig = j[0]
    label = j[1]

    image = cv.resize(orig, (96, 96))
    image = image.reshape(-1, 96, 96, 3) / 255.0
    preds = model.predict(image)
    pred_class = np.argmax(preds)

    true_label = f'Pokemon da Imagem: {classesselecionadas[label]}'
    pred_label = f'Previsto: {classesselecionadas[pred_class]} {round(preds[0][pred_class] * 100, 2)}%'

    fig.add_subplot(rows, cols, i + 1)
    plt.imshow(orig[:, :, ::-1])
    plt.title(f'{true_label}\n{pred_label}')
    plt.axis('off')
    plt.show()

plt.tight_layout()
