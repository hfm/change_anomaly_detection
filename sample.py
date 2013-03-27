from changefinder import change_finder
from numpy.random import rand, multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


## generating sample data
data_a = multivariate_normal([10.0], np.eye(1) * 0.1, 400)
data_b = multivariate_normal(rand(1) * 100, np.eye(1), 100)
X = np.r_[data_a, data_b][:, 0]

## scoring
c_cf = change_finder(term=50, window=5, order=(1, 1, 0))
result = c_cf.main(X)

## plot
fig = plt.figure()
axL = fig.add_subplot(111)
line1, = axL.plot(X, "b-", alpha=.7)
plt.ylabel("Values")

ax = plt.gca()
ax.yaxis.grid(False)
ax.xaxis.grid(True)
for tick in ax.yaxis.get_major_ticks():
    tick.tick2On = False
    tick.label2On = False
for tick in ax.xaxis.get_major_ticks():
    tick.tick2On = False
    tick.label2On = False

axR = fig.add_subplot(111, sharex=axL, frameon=False)
line2, = axR.plot(result, "r-", alpha=.7)
axR.yaxis.tick_right()
axR.yaxis.set_label_position("right")
plt.ylabel("Score")
plt.ylim(ymin=-5.0)
plt.xlabel("Sample data")

ax = plt.gca()
ax.yaxis.grid(False)
ax.xaxis.grid(False)
for tick in ax.yaxis.get_major_ticks():
    tick.tick2On = True
    tick.label2On = True
for tick in ax.xaxis.get_major_ticks():
    tick.tick2On = False
    tick.label2On = False

plt.title("Sample: Change Anomaly Detection")
plt.legend([line1, line2], ["Data", "Score"], loc=2)
plt.savefig("sample.png", dpi=144)
