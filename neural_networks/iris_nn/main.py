#!/usr/bin/env python3

# Класификация датасета iris
# Сделано на основе данной статьи:
# https://towardsdatascience.com/neural-network-on-iris-data-4e99601a42c8

from NeuralNetwork import *

iris = pd.read_csv('../../files/iris.csv')
iris.loc[iris['species'] == 'virginica', 'species'] = 0
iris.loc[iris['species'] == 'versicolor', 'species'] = 1
iris.loc[iris['species'] == 'setosa', 'species'] = 2
iris = iris[iris['species'] != 2]
print(iris)

X = iris[['petal_length', 'petal_width']].values.T
Y = iris[['species']].values.T
Y = Y.astype('uint8')

plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

nn = NeuralNetwork(X, Y, hidden=6, epochs=10000)
nn.plot(lambda x: nn.predict(x.T), X, Y[0, :])
