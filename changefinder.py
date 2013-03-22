#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy as sp
import statsmodels.tsa.arima_model as ar


## Change Finder
class change_finder(object):
    ## Costructor
    def __init__(self, term=70, window=5, order=(1, 1, 0)):
        ## @brief Quantity for learning
        self.term = term
        ## @brief Smoothing width
        self.window = window
        ## @brief Order for ARIMA model
        self.order = order
        print("term:", term, "window:", window, "order:", order)

    ## Main Function
    # @param[in] X Data Set
    # @return score vector
    def main(self, X):
        req_length = self.term * 2 + self.window + np.round(self.window / 2) - 2
        if len(X) < req_length:
            sys.exit("ERROR! Data length is not enough.")

        print("Scoring start.")
        # X = (X - np.mean(X)) / np.std(X)  # data normalized
        score = self.outlier(X)
        score = self.changepoint(score)

        space = np.zeros(len(X) - len(score))
        score = np.r_[space, score]
        print("Done.")

        return score

    ## Calculate Outlier Score from Data
    # @param[in] X Data Set
    # @return Outlier-score (M-term)-vector
    def outlier(self, X):
        count = len(X) - self.term - 1

        ## train
        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        fit = [ar.ARIMA(trains[t], self.order).fit(disp=0) for t in range(count)]

        ## predict
        resid = [fit[t].forecast(1)[0][0] - target[t] for t in range(count)]
        pred = [fit[t].predict() for t in range(count)]
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        ## logloss
        score = -sp.stats.norm.logpdf(resid, m, s)

        ## smoothing
        score = np.convolve(score, np.ones(self.window) / self.window, 'valid')

        return score

    ## Calculate ChangepointScore from OutlierScore
    # @param[in] X Data Set(Outlier Score)
    # @return Outlier-score (M-term)-vector
    def changepoint(self, X):
        count = len(X) - self.term - 1

        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        m = np.mean(trains, axis=1)
        s = np.std(trains, axis=1)

        score = -sp.stats.norm.logpdf(target, m, s)

        w = np.round(self.window / 2)
        score = np.convolve(score, np.ones(w) / w, 'valid')  # smoothing

        return score


def sample():
    data_a = sp.random.multivariate_normal([20.0, 10.0], sp.eye(2) * 2.0, 200)[:, 0]
    data_b = sp.random.multivariate_normal([-50.0, 0.0], sp.eye(2) * 3.0, 200)[:, 0]
    data_c = sp.random.multivariate_normal([50.0, 30.0], sp.eye(2) * -2.0, 200)[:, 0]
    X = np.r_[data_a, data_b, data_c]
    c_cf = change_finder(term=50, window=5)
    result = c_cf.main(X)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    axL = fig.add_subplot(111)
    line1, = axL.plot(X, "r-")
    plt.ylabel("Sample data")

    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
        tick.label2On = False

    axR = fig.add_subplot(111, sharex=axL, frameon=False)
    line2, = axR.plot(result, "g-", lw=2)
    axR.yaxis.tick_right()
    axR.yaxis.set_label_position("right")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = True
        tick.label2On = True
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
        tick.label2On = False

    plt.title("Sample: Change Anomaly Detection")

    ## title, legend
    lines = [line1, line2]
    labels = ["Data", "Score"]
    plt.legend(lines, labels, loc=2)

    plt.show()

if __name__ == '__main__':
    sample()
