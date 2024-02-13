#!/usr/bin/env python3

# Описание класса нейронной сети

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(self, X, Y, hidden, epochs=10000):
        self.a1 = None
        self.a2 = None
        self.z1 = None
        self.z2 = None
        self.hid = hidden
        self.inp = X.shape[0]
        self.out = Y.shape[0]

        np.random.seed(2)
        self.b1 = np.zeros(shape=(self.hid, 1))
        self.b2 = np.zeros(shape=(self.out, 1))
        self.w1 = np.random.randn(self.hid, self.inp) * 0.01
        self.w2 = np.random.randn(self.out, self.hid) * 0.01

        for i in range(0, epochs):
            self.forwardprop(X)
            self.backprop(X, Y)
            self.update()

    def forwardprop(self, X):
        self.z1 = np.dot(self.w1, X) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))

    def backprop(self, X, Y):
        m = X.shape[1]
        self.dz2 = self.a2 - Y
        self.dw2 = (1 / m) * np.dot(self.dz2, self.a1.T)
        self.db2 = (1 / m) * np.sum(self.dz2, axis=1, keepdims=True)
        self.dz1 = np.multiply(np.dot(self.w2.T, self.dz2), 1 - np.power(self.a1, 2))
        self.dw1 = (1 / m) * np.dot(self.dz1, X.T)
        self.db1 = (1 / m) * np.sum(self.dz1, axis=1, keepdims=True)

    def update(self, lrate=1.2):
        self.w1 = self.w1 - lrate * self.dw1
        self.w2 = self.w2 - lrate * self.dw2
        self.b1 = self.b1 - lrate * self.db1
        self.b2 = self.b2 - lrate * self.db2

    def predict(self, X):
        self.forwardprop(X)
        predictions = np.round(self.a2)
        return predictions

    def plot(self, mod, X, y):
        h = 0.01
        x_min, x_max = X[0, :].min() - 0.25, X[0, :].max() + 0.25
        y_min, y_max = X[1, :].min() - 0.25, X[1, :].max() + 0.25
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = mod(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
        plt.title("Decision Boundary for hidden layer size " + str(6))
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.show()
