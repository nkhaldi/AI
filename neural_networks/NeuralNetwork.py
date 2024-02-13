#!/usr/bin/env python3

# Описание класса нейронной сети
# Neural network class discribtion


import matplotlib.pyplot as plt
import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hidennodes, outputnodes, learning_rate):
        # Кол-во нейронов в каждом слое
        self.inodes = inputnodes
        self.hnodes = hidennodes
        self.onodes = outputnodes

        # Скорость обучения
        self.lr = learning_rate
        # wih - матрица с коэффциентами связи между входным и скрытым слоем
        # who - матрица с коэффциентами связи между скрытым и выходным слоем
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        # Другой вариант весовых коэф - нормальное распределение
        # self.wih = np.random.normal(0.0, pow(self.hnodes), -0.5)
        # self.who = np.random.normal(0.0, pow(self.onodes), -0.5)

        # Функция активации
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        # Преобразование входной список в двумерный массив
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # Прямое распространение
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Находим ошибку и обновляем веса сети
        outputs_error = targets - final_outputs
        self.who += self.lr * np.dot((outputs_error * final_outputs * (1 - final_outputs)), hidden_outputs.T)

        # Вычисление ошибок скрытого слоя
        hidden_error = np.dot(self.who.T, outputs_error)
        self.wih += self.lr * np.dot((hidden_error * hidden_outputs * (1 - hidden_outputs)), inputs.T)

    def query(self, inputs_list):
        # Преобразование входной список в двумерный массив
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
