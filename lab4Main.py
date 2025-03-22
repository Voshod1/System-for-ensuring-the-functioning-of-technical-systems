import numpy as np
import matplotlib.pyplot as plt
from math import cos, acos, sin
np.set_printoptions(linewidth=np.inf)
from functools import reduce


def setData(lenx, leny, file_x, file_y):
    # x = []
    # y = []
    # readX = open("data/" + file_x + ".txt", 'r')
    # readY = open("data/" + file_y + ".txt", 'r')
    # print(file_x)
    # for i in range(lenx):
    #    x.append(list(map(lambda z: float(z), (readX.readline()).split())))
    #    y.append(list(map(lambda z: float(z), (readY.readline()).split())))
    #    x[i] = x[i][:len(x[i])]
    #    y[i] = y[i][:len(y[i])]
    x = np.loadtxt("data/" + file_x + ".txt", max_rows=lenx)
    y = np.loadtxt("data/" + file_y + ".txt", max_rows=leny)
    # return [np.array(x), np.array(y)]
    return [x, y]


def normalize(data, isMulti):
    normalizer = []
    for arr in data:
        normalizer = []
        for i in range(arr.shape[1]):
            arr[:, i] += np.random.normal(0, (max(arr[:, i]) - min(arr[:, i])) * 0.001, len(arr[:, i]))
            m = min(arr[:, i])
            M = max(arr[:, i])
            normalizer.append([m, M])
            arr[:, i] -= m
            arr[:, i] /= (M - m)
    if (isMulti):
        data[1] += 2
        data[1] = np.log(data[1])
    return data, normalizer


def denormalize(data, normalizer):
    for arr in data:
        for i in range(len(arr)):
            arr[i] *= (normalizer[1] - normalizer[0])
            arr[i] += normalizer[0]
    return data


def setVariables(orders, numbers, x):
    variables = []

    for k in range(x.shape[0]):
        variables.append([])
        for j in range(0, numbers[0]):
            if (j == 0):
                variables[k].append(1)
            for i in range((orders[0])):
                variables[k].append(x[k][j] ** (i + 1))
        for j in range(numbers[0], numbers[1] + numbers[0]):
            for i in range((orders[1])):
                variables[k].append(x[k][j] ** (i + 1))
        for j in range(numbers[1] + numbers[0], numbers[1] + numbers[0] + numbers[2]):
            for i in range((orders[2])):
                variables[k].append(x[k][j] ** (i + 1))

    variables = np.array(variables)
    return variables


def setPolynomials(kind, orders, numbers, x, isMulti):
    if (kind == 'Чебишова 2'):
        # print("cheb")
        apply = chebyshev
    if (kind == 'Лежандра'):
        # print("lege")
        apply = legendre
    if (kind == 'Ерміта'):
        # print("her")
        apply = hermit
    if (kind == 'Лагерра'):
        # print("HI!!!!")
        # print("lagger")
        apply = laguerre
    if (kind == 'Чебишова'):
        apply = chebyshev
    polynomial = []
    for k in range(x.shape[0]):
        polynomial.append([])
        for j in range(0, numbers[0]):
            if (j == 0):
                polynomial[k].append(apply(0, 2 * x[k][j] - 1))
            for i in range((orders[0])):
                if (isMulti):
                    # print(apply(i + 1, 2*x[k][j] - 1))
                    polynomial[k].append(np.log(2 + apply(i + 1, 2 * x[k][j] - 1)))
                else:
                    polynomial[k].append(apply(i + 1, 2 * x[k][j] - 1))
        for j in range(numbers[0], numbers[1] + numbers[0]):
            for i in range((orders[1])):
                if (isMulti):
                    polynomial[k].append(np.log(2 + apply(i + 1, 2 * x[k][j] - 1)))
                else:
                    polynomial[k].append(apply(i + 1, 2 * x[k][j] - 1))
        for j in range(numbers[1] + numbers[0], numbers[1] + numbers[0] + numbers[2]):
            for i in range((orders[2])):
                if (isMulti):
                    polynomial[k].append(np.log(2 + apply(i + 1, 2 * x[k][j] - 1)))
                else:
                    polynomial[k].append(apply(i + 1, 2 * x[k][j] - 1))
    polynomial = np.array(polynomial)
    return polynomial


def setPsy(coefc, numbers, orders, x, isMulti):
    polynomial = []
    for k in range(x.shape[0]):
        polynomial.append([1])
        index = 0
        for j in range(0, numbers[0]):
            if (j == 0):
                index += 1
            else:
                index += orders[0]
            if (isMulti):
                polynomial[k].append(
                    np.log(2 + predict(coefc[index: index + orders[0]], x[k, index: index + orders[0]], isMulti)))
            else:
                polynomial[k].append(predict(coefc[index: index + orders[0]], x[k, index: index + orders[0]], isMulti))
        for j in range(0, numbers[1]):
            if (j == 0):
                index += orders[0]
            else:
                index += orders[1]
            if (isMulti):
                polynomial[k].append(
                    np.log(2 + predict(coefc[index: index + orders[1]], x[k, index: index + orders[1]], isMulti)))
            else:
                polynomial[k].append(predict(coefc[index: index + orders[1]], x[k, index: index + orders[1]], isMulti))
        for j in range(0, numbers[2]):
            if (j == 0):
                index += orders[1]
            else:
                index += orders[2]
            if (isMulti):
                polynomial[k].append(
                    np.log(2 + predict(coefc[index: index + orders[2]], x[k, index: index + orders[2]], isMulti)))
            else:
                polynomial[k].append(predict(coefc[index: index + orders[2]], x[k, index: index + orders[2]], isMulti))
    polynomial = np.array(polynomial)
    return polynomial


def setF(coefc, numbers, x, isMulti):
    polynomial = []
    for k in range(x.shape[0]):
        polynomial.append([1])
        index = 1
        for j in range(0, len(numbers)):
            if (isMulti):
                polynomial[k].append(
                    np.log(2 + predict(coefc[index: index + numbers[j]], x[k, index: index + numbers[j]], isMulti)))
            else:
                polynomial[k].append(
                    predict(coefc[index: index + numbers[j]], x[k, index: index + numbers[j]], isMulti))
            index += numbers[j]
    polynomial = np.array(polynomial)
    return polynomial


def fit(polynomial, y):
    polynomial[np.where(abs(polynomial >= 1E308))] = 1
    polynomial[np.isnan(polynomial)] = 0
    coefs = np.linalg.lstsq(polynomial, y)
    return coefs[0]


def predict(coefc, x, isMulti):
    if (isMulti):
        value = np.exp(x @ np.array(coefc).T)
    else:
        value = x @ np.array(coefc).T
    return value


def error(y, x, coefc, isMulti):
    if (isMulti):
        value = np.exp(x @ np.array(coefc).T)
        difference = np.exp(y) - value
    else:
        difference = x @ np.array(coefc).T - y
    return np.linalg.norm(difference)


def precision(y, x, prec, kind, numbers, isMulti, initial=(0, 0, 0)):
    value = prec + 1
    i, j, k = initial
    counter = 0
    while (value > prec):
        counter += 1
        value = 0
        i += 1 if (counter % 3 == 1) else 0
        j += 1 if (counter % 3 == 2) else 0
        k += 1 if (counter % 3 == 0) else 0
        pol = setPolynomials(kind, [i, j, k], numbers, x, isMulti)
        for t in range(y.shape[1]):
            lambd = fit(pol, y[:, t - 1])
            value += error(y[:, t - 1], pol, lambd, isMulti)
    return (i, j, k)


def chebyshev(n, x):
    return cos(n * acos((x)))


def chebyshev2(n, x):
    if (x == 1):
        return chebyshev2(n, x - 0.01)
    else:
        return sin((n + 1) * acos(x)) / sin(acos(x))


def legendre(n, x):
    if (n == 0):
        return 0.5
    if (n == 1):
        return x
    else:
        return (2 * n + 1) / (n + 1) * x * legendre(n - 1, x) - n / (n + 1) * legendre(n - 2, x)


def hermit(n, x, isProbabble=True):
    coef = 2 if isProbabble else 1
    if (n == 0):
        return 0.5
    if (n == 1):
        return coef * x
    else:
        return coef * (x * hermit(n - 1, x, isProbabble) - (n - 1) * hermit(n - 2, x, isProbabble))


def laguerre(n, x):
    # print(n, x)
    if (n == 0):
        return 0.5
    if (n == 1):
        return 1 - x
    else:
        return 1 / n * ((2 * n - 1 - x) * laguerre(n - 1, x) - (n - 1) * laguerre(n - 2, x))


def showAppr(kind, orders, numbers, coefc, isMulti):
    if(kind == 'Чебишова'):
        pol = 'U'
    if(kind == 'Лежандра'):
        pol = 'Leg'
    if(kind == 'Ерміта'):
        pol = 'H'
    if(kind == 'Лагерра'):
        pol = 'Lag'
    SUB = str.maketrans("0123456789", "0123456789")
    k = 0
    string = ''
    for j in range(0, numbers[0]):
        if(j==0):
            k += 1
            if(not isMulti):
                string += str(round(coefc[k], 5))
        for i in range((orders[0])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            if(not isMulti):
                string += sign + str(round(coefc[k], 5)) + str((pol + str(i + 1) +'(x' + str(j + 1) + ')')).translate(SUB)
            else:
                string += "(" + "1 " + sign + str(round(coefc[k], 5)) + (pol + str(i + 1) +'(x' + str(j + 1) + ')').translate(SUB) + ")"
            k += 1
    for j in range(numbers[0], numbers[1] + numbers[0]):
        for i in range((orders[1])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            if(not isMulti):
                string += sign + str(round(coefc[k], 5)) + str((pol + str(i + 1) +'(x' + str(j + 1) + ')')).translate(SUB)
            else:
                string += "(" + "1 " + sign + str(round(coefc[k], 5)) + (pol + str(i + 1) +'(x' + str(j + 1) + ')').translate(SUB) + ")"
            k += 1
    for j in range(numbers[1] + numbers[0], numbers[1] + numbers[0] + numbers[2]):
        for i in range((orders[2])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            if(not isMulti):
                string += sign + str(round(coefc[k], 5)) + str((pol + str(i + 1) +'(x' + str(j + 1) + ')')).translate(SUB)
            else:
                string += "(" + "1 " + sign + str(round(coefc[k], 5)) + (pol + str(i + 1) +'(x' + str(j + 1) + ')').translate(SUB) + ")"
            k += 1
    return string

def showVariables(orders, numbers, coefc):
    SUB = str.maketrans("0123456789", "0123456789")
    k = 0
    string = ''
    for j in range(0, numbers[0]):
        if(j==0):
            string += str(round(coefc[k], 5))
            k += 1
        for i in range((orders[0])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            string += sign + str(round(coefc[k], 5)) + 'x' + (str(j + 1)).translate(SUB) + '^' + str(str(i + 1))
            k += 1
    for j in range(numbers[0], numbers[1] + numbers[0]):
        for i in range((orders[1])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            string += sign + str(round(coefc[k], 5)) + 'x' + (str(j + 1)).translate(SUB) + '^' + str(str(i + 1))
            k += 1
    for j in range(numbers[1] + numbers[0], numbers[1] + numbers[0] + numbers[2]):
        for i in range((orders[2])):
            if(coefc[k] < 0):
                sign = ''
            else:
                sign = '+'
            string += sign + str(round(coefc[k], 5)) + 'x' + (str(j + 1)).translate(SUB) + '^' + str(str(i + 1))
            k += 1
    return string

def showPsy(numbers, coefc, isMulti):
    string = ''
    SUB = str.maketrans("0123456789", "0123456789")
    if(not isMulti):
        string += str(round(coefc[0], 5))
    for i in range(1, sum(numbers) + 1):
        if(coefc[i] < 0):
            sign = ''
        else:
            sign = '+'
        if(not isMulti):
            string += sign + str(round(coefc[i], 5)) + ('psi' + str(i) +'(x' + str(i) + ')').translate(SUB)
        else:
            string += "(" + "1 " + sign + str(round(coefc[i], 5)) + ('psi' + str(i) +'(x' + str(i) + ')').translate(SUB) + ")"
    return string

def showF(numbers, coefc, isMulti):
    string = ''
    SUB = str.maketrans("0123456789", "0123456789")
    if(not isMulti):
        string += str(round(coefc[0], 5))
    for i in range(1, len(numbers) + 1):
        if(coefc[i] < 0):
            sign = ''
        else:
            sign = '+'
        if(not isMulti):
            string += sign + str(round(coefc[i], 5)) + ('Fi' + str(i) +'(x' + str(i) + ')').translate(SUB)
        else:
            string += "(" + "1 " + sign + str(round(coefc[i], 5)) + ('Fi' + str(i) +'(x' + str(i) + ')').translate(SUB) + ")"
    return string


def visualize(y, x, coefc, t, normalizer, isMulti):
    plt.figure(figsize=(5.5,5.5))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    n = np.arange(0, len(y), 1)
    if(isMulti):
        forecast = np.exp(x @ np.array(coefc).T) - 2
        real = np.exp(y) - 2
        real, forecast = denormalize((real, forecast), normalizer)
    else:
        forecast = x @ np.array(coefc).T
        real, forecast = denormalize((y, forecast), normalizer)
    plt.plot(n, forecast)
    plt.plot(n, real)
    #text = plt.annotate(str(error(y, x, coefc, isMulti)), xy = [1, 1])
    #text.set_fontsize(30)
    plt.title(error(y, x, coefc, isMulti))
    plt.savefig('plot' + str(t + 1) + '.png', orientation = 'landscape')


import copy
from matplotlib import animation
from IPython.display import display, clear_output
import sys, os, random
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation


def form_data_animation(x, y, settings, build_window, forecast_window):
    j = 0
    y2 = copy.copy(y)
    y1 = np.array([])
    for i in range(len(y2) - build_window):
        if (i == j * forecast_window):
            pol = setPolynomials('Чебишова', settings[0], settings[1],
                                 x[j * forecast_window: j * forecast_window + build_window], False)
            lambd = fit(pol, y2[j * forecast_window: j * forecast_window + build_window])  # j * build_window
            # if(i == 0):
            # print(lambd)
            # print(y2[j * forecast_window: j * forecast_window + build_window])
            #    y1 = np.hstack((y1, predict(lambd, pol, False)))
            # print(x[j * forecast_window + build_window: min(len(x),(j+1) * forecast_window + build_window)])
            predicted = predict(lambd, setPolynomials('Чебишова', settings[0], settings[1], x[
                                                                                            j * forecast_window + build_window: min(
                                                                                                len(x), (
                                                                                                            j + 1) * forecast_window + build_window)],
                                                      False), False)

            # y_predicted = predict(lambd, setPolynomials('Чебишова', [1, 1, 1], [2, 2, 2],
            #                                            x[(j + 1) * build_window: (j + 1) * build_window + forecast_window], False), False)
            y1 = np.hstack((y1, predicted))
            j += 1

    return y1, y2[build_window:]


def get_mean(y):
    if len(y) == 0: return 0
    return sum(y) / len(y)


def get_variance(y):
    y1 = copy.copy(y)
    y_mean = get_mean(y1)
    y1 -= y_mean
    return sum(y1 ** 2)


def crazylation(y, y_forecast):
    y_mean = get_mean(y)
    forecast_mean = get_mean(y_forecast)
    y_centralized = y - y_mean
    y_forecast_centralized = y_forecast - forecast_mean
    result = y_centralized @ y_forecast_centralized
    correlation = np.sqrt(get_variance(y) * get_variance(y_forecast))
    if correlation == 0:
        return 0
    return result / correlation


from itertools import combinations


# list(combinations(np.arange(3),2))
def indicator(y, window):
    combinat = list(combinations(np.arange(len(y)), 2))
    indicator_fail = np.zeros((len(combinat), len(y[0]) // window - 1))
    for i in range(len(y[0]) // window - 2):
        for j in range(len(combinat)):
            if crazylation(y[combinat[j][0]][i * window:(i + 1) * window],
                           y[combinat[j][1]][(i) * window:(i + 1) * window]) * \
                    crazylation(y[combinat[j][0]][(i + 1) * window:(i + 2) * window],
                                y[combinat[j][1]][(i + 1) * window:(i + 2) * window]) < -1 * (2 / (window ** 1.15)):
                indicator_fail[j][i] = 1
                # if i*window > 600 and i*window < 800:
                # print(crazylation(y[combinat[j][0]][i*window:(i+1)*window], y[combinat[j][1]][(i)*window:(i+1)*window]) * \
                #    crazylation(y[combinat[j][0]][(i+1)*window:(i+2)*window], y[combinat[j][1]][(i+1)*window:(i+2)*window]))
                # print(i,' ',(i+1)*window)
    indicator = np.zeros(len(y[0]) // window - 1)
    for i in range(len(y[0]) // window - 1):
        tmp = 0
        for j in range(len(combinat)):
            tmp += indicator_fail[j][i]
        if tmp > 0:
            indicator[i] = 1
    return indicator


x1, y3 = setData(1000, 1000, 'x', 'y')
data, normalizerr = normalize((x1, y3), False)
x, y = data[0], data[1]
# print(y.shape)
# for i in range(40,80,1):
# print(np.array(indicator(y.T,i)))

# setting equal size of build_window and forecast_window will make forecast don't change dynamically, but update.
"""dis_y1, dis_y2 = form_data_animation(x, y,[[1,2,1],[2,2,2]] ,10, 10)"""


class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

    def compute_initial_figure(self):
        pass


class AnimationWidget(QtWidgets.QWidget):

    def __init__(self, y_init, y_real, window, autoplay, filenumber, anim_speed, danger_levels, stop_func, start_func,
                 result_table, num_of_y, normalizer, read_risks, detectors, datchicks):
        # self.func_anim = func
        QMainWindow.__init__(self)
        self.playing = autoplay
        vbox = QVBoxLayout()
        self.canvas = MyMplCanvas(self, width=5, height=4, dpi=100)
        vbox.addWidget(self.canvas)
        hbox = QHBoxLayout()
        self.start_button = QPushButton("start", self)
        self.stop_button = QPushButton("stop", self)
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)
        hbox.addWidget(self.start_button)
        hbox.addWidget(self.stop_button)
        vbox.addLayout(hbox)
        self.setLayout(vbox)

        self.datchicks = datchicks
        self.detectors = detectors
        self.normalizer = normalizer
        self.num_of_y = num_of_y
        self.read_risks = read_risks
        self.table = result_table
        self.stop_func = stop_func
        self.start_func = start_func
        self.window = window
        self.levels = danger_levels
        self.x = np.arange(len(y_init))
        self.y_fcast = y_init
        self.x1 = np.arange(len(y_real))
        self.y_real = y_real + np.random.normal(0, 0.03, len(y_real))
        self.min_val = np.ones((len(y_real))) * danger_levels[0][0]
        self.max_val = np.ones((len(y_real))) * danger_levels[1][0]
        self.correl = np.zeros((len(y_real)))
        self.filenumber = filenumber
        self.riskLev = []
        risks = abs(self.y_real[1:] - self.y_real[:-1])
        differences = np.vstack((abs(self.y_real - self.min_val), abs(self.y_real - self.max_val)))
        differences = np.array(
            list(map(lambda j: min(differences[0][j], differences[1][j]), range(differences.shape[1]))))
        risk_maxs = np.array(list(
            map(lambda j: max(risks[max(0, j - window):max(len(self.y_real), j + int(window / 2))]),
                range(len(risks)))))
        self.risk_management = (differences[1:] / risk_maxs) * self.window
        self.risk_exist = np.zeros(len(self.risk_management))
        # print(self.datchicks)
        if not read_risks:
            self.canvas.axes.set_ylim([min(self.y_real) * 0.9, max(self.y_real) * 1.1])
            for j in range(int(window / 2) + 5, len(self.risk_management) - int(window / 2)):
                if (get_variance(self.risk_management[j:j + int(window / 2)]) > 1.5 * get_variance(
                        self.risk_management[j - window:j]) or
                    abs(get_mean(self.risk_management[j:j + int(window / 2)]) - get_mean(
                        self.risk_management[j - window:j])) > 2 * get_variance(
                            self.risk_management[j - window:j]) ** 0.5) or (
                        self.filenumber == 1 and self.y_real[j] < self.min_val[0]):
                    if self.risk_exist[j - 1] == min(self.y_real) or self.risk_exist[j - 1] == 0:
                        self.risk_exist[j] = max(self.y_real)
                    else:
                        self.risk_exist[j] = min(self.y_real) + 0.01
            file = open("data/risks" + str(filenumber) + ".txt", 'w+')
            string = str(list(self.risk_management)).replace('[', '')
            string = string.replace(']', '')
            string = string.replace(',', '')
            file.write(string)
            file.close()

        else:
            self.risk_management = np.arange(len(risk_maxs))
            read_risks = np.zeros((self.num_of_y, len(self.risk_management)))
            for i in range(self.num_of_y):
                file = open("data/risks" + str(i) + ".txt", 'r')
                readl = file.readline()
                # print(readl.strip().split())
                read_risks[i] = np.array(list(map(lambda z: float(z), readl.strip().split())))
                file.close()
            for i in range(len(self.risk_management)):
                self.risk_management[i] = min(read_risks[:, i])
                # print(read_risks[:,i])
                # print(min(read_risks[:,i]))
                # print()
            self.risk_management = (self.risk_management - min(self.risk_management)) / (
                        max(self.risk_management) - min(self.risk_management))
            self.canvas.axes.set_ylim([min(self.risk_management) * 0.9, max(self.risk_management) * 1.1])
            self.canvas.axes.set_ylim([0, max(self.risk_management)*1.1])
            #print(self.risk_management)
            self.canvas.axes.set_title("Risk")

        # self.canvas.axes.set_xlim([window, len(y_init)])
        # self.canvas.axes.set_ylim([min(min(self.y_real),min(self.risk_management))*0.9, max(max(self.risk_management), max(self.y_real))*1.1])
        # self.canvas.axes.set_ylim([min(self.y_real)*0.9, max(self.y_real)*1.1])
        # risks_adequate = np.array(list(map(lambda i: min(risks[0, i], risks[1, i]), range(risks.shape[1]))))
        self.criteria, = self.canvas.axes.plot(self.x1[:1], self.risk_management[:1], animated=True, lw=1, color='red')
        f = open("data/history" + str(filenumber) + ".txt", 'r')
        readl = f.readline()
        self.data = np.array(list(map(lambda z: float(z), readl.strip().split())))
        self.y_fcast = np.append(self.data, self.y_fcast[len(self.data):])
        f.close()
        self.begin = 0
        self.index = len(self.data)
        self.i = 0
        self.line_crazy, = self.canvas.axes.plot(self.x[:1], self.correl[:1], animated=True, lw=1, color='grey')
        self.line_risk, = self.canvas.axes.plot(self.x1[:1], self.risk_exist[:1], animated=True, lw=2, color='grey')
        self.line1, = self.canvas.axes.plot(self.x1, self.y_real, animated=True, lw=1, color='black')
        self.line, = self.canvas.axes.plot(self.x, self.y_fcast, animated=True, lw=1, color='orange')
        self.line2, = self.canvas.axes.plot(self.x1, self.min_val, animated=True, lw=1, color='blue')
        self.line3, = self.canvas.axes.plot(self.x, self.max_val, animated=True, lw=1, color='green')

        self.ani = animation.FuncAnimation(
            self.canvas.figure,
            self.update_line,
            blit=True, interval=anim_speed
        )

    def update_line(self, i):

        def alarm(level, variance, growth):
            if ((self.correl[min(len(self.y_fcast) - 1, self.index)] > 0.5 and
                 self.levels[level - 1][1] - self.y_real[min(len(self.y_fcast) - 1, self.index)] < np.sqrt(
                        variance) and growth > 0)
                    or
                    (self.correl[min(len(self.y_fcast) - 1, self.index)] > 0.5 and
                     -self.levels[level - 1][0] + self.y_real[min(len(self.y_fcast) - 1, self.index)] < np.sqrt(
                                variance) and growth < 0)
                    or
                    (self.correl[min(len(self.y_fcast) - 1, self.index)] < -0.5 and
                     self.levels[level - 1][1] - self.y_real[min(len(self.y_fcast) - 1, self.index)] < np.sqrt(
                                variance) and growth < 0)
                    or
                    (self.correl[min(len(self.y_fcast) - 1, self.index)] < -0.5 and
                     -self.levels[level - 1][0] + self.y_real[min(len(self.y_fcast) - 1, self.index)] < np.sqrt(
                                variance) and growth > 0)):
                # self.func_anim("Ostorozhno -!")
                return level
            else:
                return 0

        def stop_anim():
            self.playing = True
            self.stop_button.click()
            self.ani.event_source.stop()

        def start_anim():
            self.playing = False
            self.start_button.click()
            self.ani.event_source.start()

        def find_in_column(column_num, value):
            for rowIndex in range(self.table.rowCount()):
                twItem = self.table.item(rowIndex, column_num)
                if twItem.text() == value:
                    return rowIndex
            else:
                return -1

        try:

            if self.playing and self.index < len(self.y_real):
                self.stop_func.clicked.connect(stop_anim)
                # if self.risk_exist[self.index] != 0:

                # a = copy.copy(self.correl)
                if self.index % self.window > self.window / 4:
                    correl = crazylation(self.y_fcast[
                                         min(len(self.data) + self.i - (len(self.data) + self.i) % self.window,
                                             len(self.y_fcast) - 1):
                                         min(len(self.y_fcast) - 1, len(self.data) + self.i)],
                                         self.y_real[
                                         min(max(0, len(self.data) + self.i - (len(self.data) + self.i) % self.window),
                                             len(self.y_fcast) - 1):
                                         min(len(self.y_fcast) - 1, len(self.data) + self.i)])
                else:
                    correl = 0
                self.correl[min(len(self.y_fcast) - 1, self.index)] = correl * (
                            self.normalizer[1] - self.normalizer[0]) + self.normalizer[0]
                moved_pos = self.y_fcast[min(len(self.data) + self.i - (len(self.data) + self.i) % self.window + 1,
                                             len(self.y_fcast) - 1): min(self.index, len(self.y_fcast))]
                moved_neg = self.y_fcast[min(len(self.data) + self.i - (len(self.data) + self.i) % self.window,
                                             len(self.y_fcast) - 1): min(max(0, self.index - 1), len(self.y_fcast))]
                if len(moved_pos) == len(moved_neg):
                    differences = moved_pos - moved_neg
                    growth_rate = crazylation(moved_pos, moved_neg)

                else:
                    differences = moved_pos[:min(len(moved_pos), len(moved_neg))] - moved_neg[:min(len(moved_pos),
                                                                                                   len(moved_neg))]
                    growth_rate = crazylation(moved_pos[:min(len(moved_pos), len(moved_neg))],
                                              moved_neg[:min(len(moved_pos), len(moved_neg))])

                growth = get_mean(differences)
                variance = 1. / (self.window - 1) * get_variance(
                    self.y_real[max(0, len(self.data) + self.i - (len(self.data) + i) % self.window):
                                min(len(self.y_fcast) - 1, self.index)])
                # print("index = " + str(i) + "  var = " + str(np.sqrt(variance)) + "  difference = " + str(min(self.max_val[0] - self.y_real[min(len(self.y_fcast)-1,self.index)], -self.min_val[0] + self.y_real[min(len(self.y_fcast)-1,self.index)])))

                alarms = [0] * len(self.levels)
                for i in range(len(self.levels)):
                    alarms[i] = alarm(i + 1, variance, growth)
                for al in alarms:
                    if al != 0:
                        ahahhahahahahahaahha = 0

                ######################################TABLE################################################
                row_num = find_in_column(0, str(self.index))
                if row_num >= 0:
                    # self.table.insertRow(Position)
                    # Position += 1
                    # Position -= 1
                    if not self.read_risks:
                        self.table.setItem(row_num, self.filenumber + 1, QTableWidgetItem(str(self.y_real[self.index])))

                else:
                    self.table.insertRow(0)
                    self.table.setItem(0, 0, QTableWidgetItem(str(self.index)))
                    if not self.read_risks:
                        self.table.setItem(0, self.filenumber + 1, QTableWidgetItem(str(self.y_real[self.index])))
                    else:
                        self.table.setItem(0, 4, QTableWidgetItem("Safe work."))
                    for i in range(1, 7):
                        if i != self.filenumber + 1:
                            self.table.setItem(0, i, QTableWidgetItem("-"))
                    row_num = 0


                if self.risk_exist[self.index] != 0:

                    if self.table.item(row_num, 5).text() == '1' and self.table.item(row_num, 4).text()[-1] != 'я':
                        self.table.setItem(row_num, 5, QTableWidgetItem('2'))
                        self.table.setItem(row_num, 4, QTableWidgetItem('2 param emergency'))
                        self.table.setItem(row_num, 6, QTableWidgetItem(
                            self.table.item(row_num, 6).text() + '-й, ' + str(self.filenumber + 1)))

                    elif self.table.item(row_num, 5).text() == '2' and self.table.item(row_num, 4).text()[-1] != 'я':
                        self.table.setItem(row_num, 5, QTableWidgetItem('3'))
                        self.table.setItem(row_num, 4, QTableWidgetItem('3 param emergency'))
                        self.table.setItem(row_num, 6, QTableWidgetItem(
                            self.table.item(row_num, 6).text() + '-й, ' + str(self.filenumber + 1)))
                    elif self.table.item(row_num, 4).text()[-1] != 'я':
                        self.table.setItem(row_num, 5, QTableWidgetItem('1'))
                        self.table.setItem(row_num, 4, QTableWidgetItem('1 param emergency'))
                        self.table.setItem(row_num, 6, QTableWidgetItem(str(self.filenumber + 1)))

                else:
                    if self.table.item(row_num, 5).text() not in ['1', '2', '3'] and self.table.item(row_num, 4).text()[
                        -1] != 'я':
                        self.table.setItem(row_num, 5, QTableWidgetItem('0'))
                        self.table.setItem(row_num, 4, QTableWidgetItem('Safe work.'))

                if self.y_fcast[self.index] < self.levels[1][0]:
                    self.table.setItem(row_num, 4, QTableWidgetItem('Failure'))
                    self.table.setItem(row_num, 5, QTableWidgetItem('6'))

                    self.table.setItem(row_num, 6, QTableWidgetItem(str(self.filenumber + 1) + '-rd param'))

                not_yet = 0
                for i in range(self.num_of_y):
                    if self.table.item(row_num, i + 1).text() == '-':
                        not_yet = 1
                if not_yet == 0 and self.table.item(row_num, 5).text() not in ['0', '-'] and \
                        self.table.item(row_num, 6).text()[-1] != 'р':
                    self.table.setItem(row_num, 6, QTableWidgetItem(self.table.item(row_num, 6).text() + '-й параметр'))
                ######################################TABLE################################################
                if self.table.item(row_num, 5).text()=='6':
                    self.riskLev.append(6)
                else:
                    self.riskLev.append(int(self.table.item(row_num, 5).text()))

                # self.index += 1
                # self.i += 1

                # print(str(al) + "level danger")
            if not self.playing and self.begin < 1:
                self.line.set_xdata(np.arange(len(self.data)))
                self.line.set_ydata(self.data)
                self.line1.set_xdata(np.arange(len(self.data)))
                self.line1.set_ydata(self.y_real[:len(self.data)])
                self.begin += 1

            if self.begin == 1:
                self.playing = True
                self.stop_button.click()
                self.ani.event_source.stop()

            if self.index >= len(self.y_real) - 2:
                self.ani.event_source.interval = 1000

            self.start_func.clicked.connect(start_anim)

            if self.read_risks:
                # print("equals to " + str(self.datchicks[self.index]))
                if (self.datchicks[self.index] == 1):
                    # print("alarma - ha- hi- he- ho   " + str(self.index))
                    if self.index % 2 == 0:
                        self.detectors[0].setDown(True)
                        self.detectors[1].setDown(False)
                        self.detectors[2].setDown(True)
                        self.detectors[3].setDown(False)
                    else:
                        self.detectors[0].setDown(False)
                        self.detectors[1].setDown(True)
                        self.detectors[2].setDown(False)
                        self.detectors[3].setDown(True)
                else:
                    for dat in self.detectors:
                        dat.setDown(False)
                self.criteria.set_ydata(self.risk_management[:self.index])
                #self.table.item(row_num, 5).text()
                #print(self.table.item(self.index, 5).text())
                #self.criteria.set_ydata(self.riskLev[:self.index])
                self.criteria.set_xdata(self.x1[:self.index])

                self.i += 1
                if self.index < len(self.y_real) and self.playing:
                    self.index += 1
                return [self.criteria]

            else:
                # if( self.min_val[0] - self.y_fcast[max(0,len(self.data) + self.i - (len(self.data) + i)%self.window-1)] < np.sqrt(variance) or  self.min_val[0] - self.y_fcast[max(0,len(self.data) + self.i - (len(self.data) + i)%self.window-1)] < np.sqrt(variance))
                self.line_crazy.set_ydata(self.correl[:self.index])
                self.line_crazy.set_xdata(self.x1[:self.index])
                y = self.y_fcast[:min(len(self.y_fcast) - 1,
                                      len(self.data) + self.i - (len(self.data) + self.i) % self.window + self.window)]
                x = self.x[:min(len(self.y_fcast) - 1,
                                (len(self.data) + self.i) - (len(self.data) + self.i) % self.window + self.window)]
                self.line.set_ydata(y)
                self.line.set_xdata(x)
                y1 = self.y_real[:min(len(self.y_real) - 1, self.index)]
                x1 = self.x1[:min(len(self.y_real) - 1, self.index)]
                self.i += 1
                if self.index < len(self.y_real) and self.playing:
                    self.index += 1
                self.line1.set_ydata(y1)
                self.line1.set_xdata(x1)
                # self.line_crazy.set_ydata(self.correl)
                # self.func_anim(str(self.index) + '\t||\t' + str(round(correl,4)))

            return [self.line1, self.line, self.line3, self.line2, self.line_crazy]

        except IndexError:
            self.playing = False
            self.ani.event_source.stop()
            if self.read_risks:
                return [self.criteria]

            # return [self.line1, self.line,self.line3, self.line2, self.line_crazy, self.line_risk]
            else:
                return [self.line1, self.line, self.line3, self.line2]

    def on_start(self):
        if self.playing:
            pass
        else:
            self.begin += 1
            self.playing = True
            self.ani.event_source.start()

    def on_stop(self):
        if self.playing:
            string = str(list(self.y_fcast[:self.i + len(self.data)])).replace('[', '')
            string = string.replace(']', '')
            string = string.replace(',', '')
            f = open("data/history" + str(self.filenumber) + ".txt", 'w+')
            if self.index >= len(self.y_fcast) - 1:
                f.write('')
            else:
                f.write(string)
            f.close()
            self.playing = False
            self.ani.event_source.stop()
        else:
            pass

    def closeEvent(self, event):

        event.accept()


def blablablabla(y, y1, window, autoplay, file_number, animation_speed, danger_levels, stop_func, start_func,
                 result_table, num_of_y, normalizer, read_risks, detectors, datchicks):
    # print(y, y1)
    screen_rect = app.desktop().screenGeometry()
    width, height = screen_rect.width(), screen_rect.height()
    win = AnimationWidget(y, y1, window, autoplay, file_number, animation_speed, danger_levels, stop_func, start_func,
                          result_table, num_of_y, normalizer, read_risks, detectors, datchicks)
    win.setGeometry(int(25 + (file_number % 2) * 2 * width / 3), int(25 + (file_number // 2) * height / 2),
                    int(height * 0.45), int(height * 0.45))
    win.show()
    w.ui.widgetList.append(win)


def blablablabla_for_check(y, y1, window, autoplay, file_number):
    # print(y, y1)
    win = AnimationWidget(y, y1, window, autoplay, file_number)
    win.move(file_number, file_number)
    win.show()
    w.ui.widgetList.append(win)
    # qApp = QApplication(sys.argv)

    # aw = AnimationWidget(y,y1, window)
    # aw.show()
    # sys.exit(qApp.exec_())


from sys import argv
import time
from Sisan4_new import Ui_Sisan4
# from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMainWindow, QApplication


# from PyQt5.QtGui import *
def get_data(window):
    parameters = []
    parameters.append(window.ui.spinRang_x1.value())
    parameters.append(window.ui.spinRang_x2.value())
    parameters.append(window.ui.spinRang_x3.value())
    parameters.append(window.ui.spinSize_x1.value())
    parameters.append(window.ui.spinSize_x2.value())
    parameters.append(window.ui.spinSize_x3.value())
    parameters.append(window.ui.spinSize_Y.value())
    parameters.append(window.ui.spinSize.value())
    parameters.append(window.ui.radioButton_2)
    parameters.append(window.ui.radioButton_4)
    parameters.append(window.ui.radioButton_6)
    parameters.append(window.ui.radioButton_8)
    return parameters


def clear_function():
    for i in range(4):
        f = open("data/history" + str(i) + ".txt", 'w+')
        f.write('')
        f.close()
    for i in range(3):
        f = open("data/risks" + str(i) + ".txt", 'w+')
        f.write('')
        f.close()


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Sisan4()
        self.ui.setupUi(self)
        self.ui.widgetList = []
        self.show()

    def closeEvent(self, event):
        event.accept()


app = 0
app = QApplication(argv)
w = AppWindow()
w.show()
tabs = []


# window_forecast = 30
# window_build = 30
def spanish_inquisition():
    window_forecast = w.ui.spinWindow2.value()
    window_build = w.ui.spinWindow1.value()
    # w.ui.resultsText.clear()
    w.ui.remove_tabs(tabs)
    parameters = get_data(w)
    detectors = [parameters[8], parameters[9], parameters[10], parameters[11]]
    # clearButton
    speed = w.ui.spinSpeed.value()
    file_x = w.ui.lineFile_from_x.text()
    file_y = w.ui.lineFile_from_y.text()
    isMulti = w.ui.isMulti.isChecked()
    data, normalizer = normalize(setData(parameters[7], parameters[7], file_x, file_y), isMulti)
    x, y = data[0], data[1]
    numbers = [parameters[3], parameters[4], parameters[5]]
    orders = [parameters[0], parameters[1], parameters[2]]
    kind = w.ui.comboBox.currentText()
    prec = 10 ** (-w.ui.spinDelta.value())
    w.ui.clearButton.clicked.connect(clear_function)
    column_count = 7
    row_count = 0  # len(parametrs[-1])
    w.ui.resultsTable.setColumnCount(column_count)
    w.ui.resultsTable.setRowCount(row_count)
    """w.ui.resultsTable.setHorizontalHeaderLabels(
        ["Час", "У1", "У2", "У3", "Стан функціонування", "Рівень небезпеки", "Причина"])"""
    w.ui.resultsTable.setHorizontalHeaderLabels(
        ["Time", "Network", "Fuel", "Battery", "System state", "Emergency level", "Cause"])
    w.ui.resultsTable.resizeColumnsToContents()
    w.ui.resultsTable.setColumnWidth(0, 50)
    w.ui.resultsTable.setColumnWidth(1, 80)
    w.ui.resultsTable.setColumnWidth(2, 80)
    w.ui.resultsTable.setColumnWidth(3, 80)
    w.ui.resultsTable.setColumnWidth(4, 140)
    w.ui.resultsTable.setColumnWidth(6, 50)
    w.ui.resultsTable.setColumnWidth(7, 100)
    w.ui.resultsTable.horizontalHeader().setStretchLastSection(True)
    w.ui.resultsTable.insertRow(0)
    for i in range(7):
        w.ui.resultsTable.setItem(0, i, QTableWidgetItem('-'))

    if w.ui.checkBox.isChecked():
        sample = precision(y[:, : w.ui.spinSize_Y.value()], x, prec, kind, numbers, isMulti)
    # w.ui.resultsText.appendPlainText('\tОПТИМАЛЬНІ ПАРАМЕТРИ: ' + str(sample))
    arr = [[[11.7, 1e10], [11.5, 1e10]], [[4.1, 1e10], [0.5, 1e10]], [[11.85, 1e10], [11.80, 1e10]]]
    datchicks = indicator(y.T, window_forecast)
    datchicks = list(map(lambda i: [datchicks[i]] * window_forecast, range(len(datchicks))))
    datchicks = list(reduce(lambda x, y: x + y, datchicks))
    for i in range(parameters[6]):
        # time.sleep(5)
        y1, y2 = form_data_animation(x, y[:, i], [orders, numbers], window_build, window_forecast)
        y1, y2 = denormalize((y1, y2), normalizer[i])

        y2 = y2 + np.random.normal(0, 0.01 * (max(y2) - min(y2)), len(y2))

        blablablabla(y1, y2, window_forecast, True, i, speed, arr[i], w.ui.stopButton, w.ui.startButton,
                     w.ui.resultsTable, parameters[6], normalizer[i], False, [], datchicks)
        # print(y1,y2)

    blablablabla(y1, y2, window_forecast, True, 3, speed, [[-1, 400]] * 4, w.ui.stopButton, w.ui.startButton,
                 w.ui.resultsTable, parameters[6], normalizer[i], True, detectors, datchicks)

    f = open("data/result.txt", "w+")
    # f.write(beautifulPolynomial + beautifulPsy)
    f.close()


w.ui.clearButton.clicked.connect(clear_function)

w.ui.Button.clicked.connect(spanish_inquisition)
app.exec_()

# w.ui.pushButton.clicked.connect(w.safeExit)
