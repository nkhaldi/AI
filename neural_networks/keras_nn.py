#!/usr/bin/env python3

from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

(tr_images, tr_labels), (ts_images, ts_labels) = mnist.load_data()

# Подготовка исходных данных
tr_images = tr_images.reshape((60000, 28*28))
tr_images = tr_images.astype('float32') / 255
ts_images = ts_images.reshape((10000, 28*28))
ts_images = ts_images.astype('float32') / 255

# Подготовка меток
tr_labels = to_categorical(tr_labels)
ts_labels = to_categorical(ts_labels)

# Aрхитектура сети
nn = models.Sequential()
nn.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
nn.add(layers.Dense(100, activation='relu'))
nn.add(layers.Dense(10, activation='softmax'))
nn.summary()

# Компиляция модели
nn.compile(optimizer='adam',
           loss='categorical_crossentropy',
           metrics=['accuracy'])

# Обучение сети методом fit и тестирование
nn.fit(tr_images, tr_labels, epochs=10, batch_size=128)
ts_loss, ts_acc = nn.evaluate(ts_images, ts_labels)
print('ts_acc: ', ts_acc)

# Создание проверочного набора
x_val = tr_images[:5000]
partial_x_train = tr_images[5000:]
y_val = tr_labels[:5000]
partial_y_train = tr_labels[5000:]

# Обучение сети
hist = nn.fit(partial_x_train, partial_y_train,
              epochs=10, batch_size=512,
              validation_data=(x_val, y_val))
hist_dict = hist.history
hist_dict.keys()

# Визуализация
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Потери на этапе обучения')
plt.plot(epochs, val_loss, 'b', label='Потери на этапе проверки')
plt.title('Потери на этапе обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()

plt.clf()
acc_values = hist_dict['accuracy']
val_acc_values = hist_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='точность на этапе обучения')
plt.plot(epochs, val_acc, 'b', label='Точность на этапе проверки')
plt.title('Точность на этапах обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
