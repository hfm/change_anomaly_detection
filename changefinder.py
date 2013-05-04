#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import scipy as sp
import statsmodels.tsa.arima_model as ar


# Change Finder
class change_finder(object):
    # Costructor

    def __init__(self, term=70, window=5, order=(1, 1, 0)):
        # @brief Quantity for learning
        self.term = term
        # @brief Smoothing width
        self.window = window
        # @brief Order for ARIMA model
        self.order = order
        print("term:", term, "window:", window, "order:", order)

    # Main Function
    # @param[in] X Data Set
    # @return score vector
    def main(self, X):
        req_length = self.term * 2 + self.window + np.round(self.window / 2) - 2
        if len(X) < req_length:
            sys.exit("ERROR! Data length is not enough.")

        print("Scoring start.")
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        score = self.outlier(X)
        score = self.changepoint(score)

        space = np.zeros(len(X) - len(score))
        score = np.r_[space, score]
        print("Done.")

        return score

    # Calculate Outlier Score from Data
    # @param[in] X Data Set
    # @return Outlier-score (M-term)-vector
    def outlier(self, X):
        count = len(X) - self.term - 1

        # train
        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        fit = [ar.ARIMA(trains[t], self.order).fit(disp=0) for t in range(count)]

        # predict
        resid = [fit[t].forecast(1)[0][0] - target[t] for t in range(count)]
        pred = [fit[t].predict() for t in range(count)]
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        # logloss
        score = -sp.stats.norm.logpdf(resid, m, s)

        # smoothing
        score = self.smoothing(score, self.window)

        return score

    # Calculate ChangepointScore from OutlierScore
    # @param[in] X Data Set(Outlier Score)
    # @return Outlier-score (M-term)-vector
    def changepoint(self, X):
        count = len(X) - self.term - 1

        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        m = np.mean(trains, axis=1)
        s = np.std(trains, axis=1)

        score = -sp.stats.norm.logpdf(target, m, s)
        score = self.smoothing(score, np.round(self.window / 2))

        return score

    # Calculate ChangepointScore from OutlierScore
    # @param[in] X Data set
    # @param[in] w Window size
    # @return Smoothing-score
    def smoothing(self, X, w):
        return np.convolve(X, np.ones(w) / w, 'valid')


def sample():
    from numpy.random import rand, multivariate_normal
    data_a = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    data_b = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    data_c = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)

    data_d = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    X = np.r_[data_a, data_b, data_c, data_d][:, 0]
    c_cf = change_finder(term=70, window=7, order=(2, 2, 0))
    result = c_cf.main(X)
    return result


if __name__ == '__main__':
    print(sample())
