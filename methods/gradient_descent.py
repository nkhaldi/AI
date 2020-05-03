#!/usr/bin/env python3

# Метод градиентного спуск
# Gradient descent


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def fig2data(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.array(fig.canvas.renderer._renderer)
    return buf


def plot_original_data():
    df = pd.read_csv("../files/data.csv")
    plt.scatter(df['radio'], df['sales'], color='#1f77b4', marker='o')
    plt.xlabel("Radio, M$")
    plt.ylabel("Sales, Units")
    plt.title("Sales vs radio ad spendings")
    axes = plt.gca()
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 35])
    plt.show()

    # Image saver
    '''
    plt.tight_layout()
    fig1 = plt.gcf()
    nfig = fig2data(fig1)
    seq.append(nfig)

    fig1.subplots_adjust(top=0.98, bottom=0.1,
                         right=0.98, left=0.08,
                         hspace=0, wspace=0)
    fig1.savefig('../files/gradient_descent-1.png',
                 dpi=1000, bbox_inches='tight', pad_inches=0)
    '''


def update_w_and_b(spendings, sales, w, b, alpha):
    dr_dw = 0.0
    dr_db = 0.0
    N = len(spendings)
    for i in range(N):
        dr_dw += -2 * spendings[i] * (sales[i] - (w * spendings[i] + b))
        dr_db += -2 * (sales[i] - (w * spendings[i] + b))
    w = w - (dr_dw / float(N)) * alpha
    b = b - (dr_db / float(N)) * alpha
    return w, b


def loss(spendings, sales, w, b):
    N = len(spendings)
    total_error = 0.0
    for i in range(N):
        total_error += (sales[i] - (w * spendings[i] + b)) ** 2
    return total_error / N


def plot_train(ep, spendings, sales, w, b):
    plt.clf()
    plt.figure(ep + 1)
    plt.xlabel("spendings, M$")
    plt.ylabel("Sales, Units")

    axes = plt.gca()
    axes.set_xlim([0, 50])
    axes.set_ylim([0, 35])

    plt.scatter(spendings, sales, color='#1f77b4', marker='o')
    X_plot = np.linspace(0, 50, 50)
    plt.plot(X_plot, X_plot * w + b)

    rnd_loss = str(round(loss(spendings, sales, w, b)))
    heading = 'epoch: ' + str(ep) + ' loss: ' + rnd_loss
    plt.title(heading)
    plt.show()

    # Image saver
    '''
    plt.tight_layout()
    fig1 = plt.gcf()
    nfig = fig2data(fig1)
    seq.append(nfig)
    fig1.subplots_adjust(top=0.98, bottom=0.1, right=0.98, left=0.08,
                         hspace=0, wspace=0)
    '''


def train(spendings, sales, w, b, alpha, epochs):
    for ep in range(epochs):
        w, b = update_w_and_b(spendings, sales, w, b, alpha)
        print('epoch:', ep, 'loss:', loss(spendings, sales, w, b))
        print("w, b: ", w, b)
        # plot_train(ep, spendings, sales, w, b)
    return w, b


def predict(x, w, b):
    return w * x + b


if __name__ == '__main__':
    df = pd.read_csv("../files/data.csv", index_col='Unnamed: 0')
    seq = list()
    print(df)
#    plot_original_data()

    print()
    df = pd.read_csv("../files/data.csv")
    x = df['radio']
    y = df['sales']
    w, b = train(x, y, 0, 0, 0.001, 10)
    print()
    x_new = 23.0
    y_new = predict(x_new, w, b)
    print(x_new, y_new)

    # Comparing to sklearn
    '''
    rex = x.values.reshape(100, 2)
    rey = y.values.reshape(100, 2)
    model = LinearRegression().fit(rex, rey)
    x_new = 23.0
    y_new = model.predict(x_new)
    print(x_new, y_new)
    '''
