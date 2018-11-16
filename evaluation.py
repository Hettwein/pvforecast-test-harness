import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import os

dir = './test_results/'
if not os.path.exists(dir):
    os.makedirs(dir)
    

def _draw_boxplot(data, offset, ax, edge_color, fill_color, mticks=False, num=1):
    pos = np.arange(num) + offset 
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, whis='range', manage_xticks=mticks)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp
    
def draw_boxplot(m_col, p_col, l_col, method, horizon, start=None, end=None):
    fig, ax = plt.subplots(figsize=(12, 12))
    bp1 = _draw_boxplot(p_col[start:end] - m_col[start:end], -0.2, ax, 'red', 'tomato')
    bp2 = _draw_boxplot(l_col[start:end] - m_col[start:end], 0.2, ax, 'blue', 'cornflowerblue')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [horizon, method])
    name = dir + 'boxplot_' + horizon + '_vs_' + method
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    fig.savefig(name + '.png', bbox_inches='tight')
    fig.show()
    
def draw_boxplot_monthly(m_col, p_col, l_col, method, horizon):
    fig, ax = plt.subplots(figsize=(18, 18))
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12 = '2016-02-01 00:00:00', '2016-03-01 00:00:00', '2016-04-01 00:00:00', '2016-05-01 00:00:00', '2016-06-01 00:00:00', '2016-07-01 00:00:00', '2016-08-01 00:00:00', '2016-09-01 00:00:00', '2016-10-01 00:00:00', '2016-11-01 00:00:00', '2016-12-01 00:00:00', '2017-01-01 00:00:00', '2017-02-01 00:00:00'
    preds = np.array([p_col[m0:m1] - m_col[m0:m1], p_col[m1:m2] - m_col[m1:m2],
                      p_col[m2:m3] - m_col[m2:m3], p_col[m3:m4] - m_col[m3:m4],
                      p_col[m4:m5] - m_col[m4:m5], p_col[m5:m6] - m_col[m5:m6],
                      p_col[m6:m7] - m_col[m6:m7], p_col[m7:m8] - m_col[m7:m8],
                      p_col[m8:m9] - m_col[m8:m9], p_col[m9:m10] - m_col[m9:m10],
                      p_col[m10:m11] - m_col[m10:m11], p_col[m11:m12] - m_col[m11:m12]])
    pvlibs = np.array([l_col[m0:m1] - m_col[m0:m1], l_col[m1:m2] - m_col[m1:m2],
                      l_col[m2:m3] - m_col[m2:m3], l_col[m3:m4] - m_col[m3:m4],
                      l_col[m4:m5] - m_col[m4:m5], l_col[m5:m6] - m_col[m5:m6],
                      l_col[m6:m7] - m_col[m6:m7], l_col[m7:m8] - m_col[m7:m8],
                      l_col[m8:m9] - m_col[m8:m9], l_col[m9:m10] - m_col[m9:m10],
                      l_col[m10:m11] - m_col[m10:m11], l_col[m11:m12] - m_col[m11:m12]])
    bp1 = _draw_boxplot(preds, -0.2, ax, 'red', 'tomato', True, len(preds))
    bp2 = _draw_boxplot(pvlibs, 0.2, ax, 'blue', 'cornflowerblue', True, len(pvlibs))
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [horizon, method])
    ax.set_xticklabels(['february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'january'])
    fig.savefig(dir + 'boxplot_' + horizon + '_vs_' + method + '_monthly.png', bbox_inches='tight')
    fig.show()

def walkForwardDailyLoss(test_y, pred_y, method_y, method, horizon):
    j = int(len(test_y) / 24)
    d1 = np.array_split(test_y, j)
    d2 = np.array_split(pred_y, j)
    d3 = np.array_split(method_y, j)
    pred_error = pd.DataFrame([math.sqrt(mean_squared_error(d2[i], d1[i])) for i in range(len(d1))])
    method_error = pd.DataFrame([math.sqrt(mean_squared_error(d3[i], d1[i])) for i in range(len(d1))])
    print('daily mean ' + horizon + ' RMSE: ' + str(pred_error.mean()[0]))
    print('daily mean ' + method + ' forecast RMSE: ' + str(method_error.mean()[0]))
    print(pred_error.describe())
    print(method_error.describe())
    
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(pred_error)
    ax.plot(method_error)
    ax.set_title("Daily RMSE")
    ax.set_ylabel("RMSE in Watts")
    ax.set_yscale('linear')
    ax.legend([horizon + ' error', method + ' error'])
    fig.autofmt_xdate()
    plt.grid(True)
    fig.savefig(dir + horizon + '_dailyRMSE_full_' + '.png')
    fig.show()

def scatter_predictions(test_y, pred_y, horizon):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(test_y, pred_y)
    ax.plot([np.amin(test_y), np.amax(test_y)], [np.amin(pred_y), np.amax(pred_y)], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig(dir + 'scatter_' + horizon + '.png')
    fig.show()
    
def plot_timeseries(m_col, p_col, l_col, method, horizon, start=None, end=None):
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(m_col[start:end])
    ax.plot(p_col[start:end])
    if l_col is not None:
        ax.plot(l_col[start:end])
    ax.set_title("pv power prediction")
    ax.set_ylabel("Power")
    ax.set_yscale('linear')
    ax.legend(['measured', horizon, method])
    fig.autofmt_xdate()
    plt.grid(True)
    name = dir + horizon + '_timeseries'
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    if l_col is not None:
        name += '_with_' + method
    fig.savefig(name + '.png')
    fig.show()

def draw_histogram(p_col, m_col, horizon):
    fig, ax = plt.subplots(figsize=(12, 12)) 
    ax.hist(p_col - m_col, bins=200)
    fig.savefig(dir + horizon + '_histogram.png')
    fig.show()
    
def draw_history(history, test=False):
    hist = pd.DataFrame.from_dict(history.history)
    fig, ax = plt.subplots(figsize=(12, 12)) 
    ax.plot(hist)
    ax.legend(hist.columns.values)
    plt.grid(True)
    name = dir + 'loss_history'
    if test:
        name += '_test'
    fig.savefig(name + '.png')
    fig.show()