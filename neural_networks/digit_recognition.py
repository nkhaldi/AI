#!/usr/bin/env python3

# Распознование рукописных цифр
# Handwriten digits recognition

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

from NeuralNetwork import *


with open('training/mnist_train_100.csv', 'r') as f:
    data_list = f.readlines()

# asfarray - преобразует строки в числа и создаёт массив
all_values = data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys')
# plt.show()

train = data_list[13].split(',')
image_array = np.asfarray(train[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys')
# plt.show()

# Подготовка данных: значения от 0.01 до 1.0
scaled_input = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01

# Создаём сеть для распознования цифр MNIST
# Количество входных, скрытых и выходных узлов и коэффициент обучения:
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

# Создаём экземпляр нейронной сети
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Тренировка нейронной сети
# Создаём целевые выходные значения (все равны 0,01, за исключением
# желаемого маркерного значения, равного 0,99 - all_values[0]
print("Test dataset")
with open('training/mnist_train_100.csv', 'r') as f:
    training_data_list = f.readlines()
for record in training_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    nn.train(inputs, targets)

# Тестирование сети
with open('training/mnist_test_10.csv', 'r') as f:
    test_data_list = f.readlines()
all_values = test_data_list[0].split(',')
all_values = test_data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28, 28))
# plt.imshow(image_array, cmap='Greys')
# plt.show()

nn.query(np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)

# Журнал оценок работы сети, первоначально пустой
scorecard = list()
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = nn.query(inputs)

    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    print('Answer:', label, end='\t')
    print('Correct:', correct_label)
print('Efficiency:', sum(scorecard) / len(scorecard), end='\n\n')

# Тренировка на полном наборе данных
print('Full dataset')
with open('training/mnist_train.csv', 'r') as f:
    training_data_list = f.readlines()
for record in training_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    nn.train(inputs, targets)

# Тестирование сети
scorecard = list()
with open('training/mnist_test.csv', 'r') as f:
    test_data_list = f.readlines()
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
    outputs = nn.query(inputs)

    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    print('Answer:', label, end='\t')
    print('Correct:', correct_label)
print('Efficiency:', sum(scorecard) / len(scorecard), end='\n\n')

# Улучшение результатов: многократное повторение тренировочных циклов
# Тренировка нейронной сети
print('Training with repeates')
epochs = 2
for е in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        nn.train(inputs, targets)

# Тестирование сети
scorecard = list()
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = nn.query(inputs)

    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    print('Answer:', label, end='\t')
    print('Correct:', correct_label)
print('Efficiency:', sum(scorecard) / len(scorecard))
