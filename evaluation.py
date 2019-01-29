import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import os

dir = './test_results/'#_bifacial/'
figsize = (10, 10)
if not os.path.exists(dir):
    os.makedirs(dir)

def set_dir(directory):
    global dir
    dir = directory
    if not os.path.exists(dir):
        os.makedirs(dir)
    print('output directory set to ' + dir)

def _draw_boxplot(data, offset, ax, edge_color, fill_color, mticks=True, num=1, outliers=True):
    pos = np.arange(num) + offset
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, manage_xticks=mticks, sym='x', capprops={'linewidth': 2}, medianprops={'linewidth': 2}, showfliers=outliers)
    for element in ['boxes', 'whiskers', 'fliers', 'caps', 'medians']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    return bp
    
def draw_boxplot(m_col, p_col, l_col=None, method=None, horizon='', start=None, end=None, title='', unit='w', outliers=True):
    fig, ax = plt.subplots(figsize=figsize)
    if l_col is not None:
        bp1 = _draw_boxplot(p_col[start:end] - m_col[start:end], -0.2, ax, 'black', 'orange', False, outliers=outliers)
        bp2 = _draw_boxplot(l_col[start:end] - m_col[start:end], 0.2, ax, 'black', 'green', False, outliers=outliers)
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [horizon, method])
        name = dir + 'boxplot_' + horizon + '_vs_' + method
    else:
        bp1 = _draw_boxplot(p_col[start:end] - m_col[start:end], 0, ax, 'black', 'orange')
        ax.legend([bp1["boxes"][0]], [horizon])
        name = dir + 'boxplot_' + horizon
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    ax.set_title(title)
    ax.set_ylabel('Error [' + unit + ']')
    ax.set_xticklabels([''])
    plt.grid(True)
    fig.savefig(name + '.png', bbox_inches='tight')
    plt.close(fig)
    
def draw_boxplot_monthly(m_col, p_col, l_col, method, horizon, title, unit, outliers=True):
    fig, ax = plt.subplots(figsize=figsize)
    m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22 = '2016-02-01 00:00:00', '2016-03-01 00:00:00', '2016-04-01 00:00:00', '2016-05-01 00:00:00', '2016-06-01 00:00:00', '2016-07-01 00:00:00', '2016-08-01 00:00:00', '2016-09-01 00:00:00', '2016-10-01 00:00:00', '2016-11-01 00:00:00', '2016-12-01 00:00:00', '2017-01-01 00:00:00', '2017-02-01 00:00:00', '2017-03-01 00:00:00', '2017-04-01 00:00:00', '2017-05-01 00:00:00', '2017-06-01 00:00:00', '2017-07-01 00:00:00', '2017-08-01 00:00:00', '2017-09-01 00:00:00', '2017-10-01 00:00:00', '2017-11-01 00:00:00', '2017-12-01 00:00:00'
    preds = np.array([p_col[m0:m1] - m_col[m0:m1], p_col[m1:m2] - m_col[m1:m2],
                      p_col[m2:m3] - m_col[m2:m3], p_col[m3:m4] - m_col[m3:m4],
                      p_col[m4:m5] - m_col[m4:m5], p_col[m5:m6] - m_col[m5:m6],
                      p_col[m6:m7] - m_col[m6:m7], p_col[m7:m8] - m_col[m7:m8],
                      p_col[m8:m9] - m_col[m8:m9], p_col[m9:m10] - m_col[m9:m10],
                      p_col[m10:m11] - m_col[m10:m11], p_col[m11:m12] - m_col[m11:m12],
                      p_col[m12:m13] - m_col[m12:m13], p_col[m13:m14] - m_col[m13:m14],
                      p_col[m14:m15] - m_col[m14:m15], p_col[m15:m16] - m_col[m15:m16],
                      p_col[m16:m17] - m_col[m16:m17], p_col[m17:m18] - m_col[m17:m18],
                      p_col[m18:m19] - m_col[m18:m19], p_col[m19:m20] - m_col[m19:m20],
                      p_col[m20:m21] - m_col[m20:m21], p_col[m21:m22] - m_col[m21:m22],
                      p_col[m22:] - m_col[m22:]])
    pvlibs = np.array([l_col[m0:m1] - m_col[m0:m1], l_col[m1:m2] - m_col[m1:m2],
                      l_col[m2:m3] - m_col[m2:m3], l_col[m3:m4] - m_col[m3:m4],
                      l_col[m4:m5] - m_col[m4:m5], l_col[m5:m6] - m_col[m5:m6],
                      l_col[m6:m7] - m_col[m6:m7], l_col[m7:m8] - m_col[m7:m8],
                      l_col[m8:m9] - m_col[m8:m9], l_col[m9:m10] - m_col[m9:m10],
                      l_col[m10:m11] - m_col[m10:m11], l_col[m11:m12] - m_col[m11:m12],
                      l_col[m12:m13] - m_col[m12:m13], l_col[m13:m14] - m_col[m13:m14],
                      l_col[m14:m15] - m_col[m14:m15], l_col[m15:m16] - m_col[m15:m16],
                      l_col[m16:m17] - m_col[m16:m17], l_col[m17:m18] - m_col[m17:m18],
                      l_col[m18:m19] - m_col[m18:m19], l_col[m19:m20] - m_col[m19:m20],
                      l_col[m20:m21] - m_col[m20:m21], l_col[m21:m22] - m_col[m21:m22],
                      l_col[m22:] - m_col[m22:]])
    bp1 = _draw_boxplot(preds, -0.2, ax, 'black', 'orange', True, len(preds), outliers)
    bp2 = _draw_boxplot(pvlibs, 0.2, ax, 'black', 'green', True, len(pvlibs), outliers)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [horizon, method])
    fig.autofmt_xdate()
    ax.set_title(title)
    ax.set_ylabel('Error [' + unit + ']')
    ax.set_xticklabels(['february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'])
    plt.grid(True)
    fig.savefig(dir + 'boxplot_' + horizon + '_vs_' + method + '_monthly.png', bbox_inches='tight')
    plt.close(fig)

def walkForwardDailyLoss(test_y, pred_y, method_y=None, method=None, horizon='1'):
    j = int(len(test_y) / 24)
    d1 = np.array_split(test_y, j)
    d2 = np.array_split(pred_y, j)
    pred_error = pd.DataFrame([math.sqrt(mean_squared_error(d2[i], d1[i])) for i in range(len(d1))])
    #print('daily mean ' + horizon + ' RMSE: ' + str(pred_error.mean()[0]))
    
    if method_y is not None:
        d3 = np.array_split(method_y, j)
        method_error = pd.DataFrame([math.sqrt(mean_squared_error(d3[i], d1[i])) for i in range(len(d1))])
        #print('daily mean ' + method + ' forecast RMSE: ' + str(method_error.mean()[0]))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(pred_error, color='orange', linewidth=2)
    if method_y is not None:
        ax.plot(method_error, color='green', linestyle='dashed', linewidth=2)
    ax.set_title('Daily RMSE')
    ax.set_ylabel('RMSE [w]')
    ax.set_xlabel('Day')
    if method_y is not None:
        ax.legend([horizon + ' error', method + ' error'])
    else:
        ax.legend([horizon + ' error'])
    fig.autofmt_xdate()
    plt.grid(True)
    fig.savefig(dir + horizon + '_dailyRMSE_full_' + '.png')
    plt.close(fig)

def scatter_predictions(test_y, pred_y, horizon):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(test_y, pred_y, marker='^')
    ax.plot([np.amin(test_y), np.amax(test_y)], [np.amin(test_y), np.amax(test_y)], 'k--', lw=3)####?[np.amin(pred_y), np.amax(pred_y)], 'k--', lw=3)####?
    ax.set_title('Power measured vs predicted')
    ax.set_xlabel('Measured power [w]')
    ax.set_ylabel('Predicted power [w]')
    plt.grid(True)
    fig.savefig(dir + 'scatter_' + horizon + '.png')
    plt.close(fig)
    
def plot_timeseries(m_col, p_col, l_col, method, horizon, start=None, end=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(p_col[start:end], color='orange', linewidth=2)
    if l_col is not None:
        ax.plot(l_col[start:end], color='green', linestyle='dashed', linewidth=2)
        ax.plot(m_col[start:end], color='blue', linestyle='dotted', linewidth=2)
        ax.legend([horizon, method, 'measured'])
    else:
        ax.plot(m_col[start:end], color='blue', linestyle='dotted', linewidth=2)
        ax.legend([horizon, 'measured'])
    ax.set_title('Power prediction')
    ax.set_ylabel('Power [w]')
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
    plt.close(fig)

def draw_histogram(p_col, m_col, horizon):
    fig, ax = plt.subplots(figsize=figsize) 
    ax.hist(p_col - m_col, bins=200)
    ax.set_title('Error distribution')
    ax.set_ylabel('Number of predictions')
    ax.set_xlabel('Error [w]')
    fig.autofmt_xdate()
    fig.savefig(dir + horizon + '_histogram.png')
    plt.close(fig)
    
def draw_history(history, test=False):
    hist = pd.DataFrame.from_dict(history.history)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hist)
    ax.set_title('Loss history')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE [w]')
    ax.legend(hist.columns.values)
    fig.autofmt_xdate()
    plt.grid(True)
    name = dir + 'loss_history'
    if test:
        name += '_test'
    fig.savefig(name + '.png')
    plt.close(fig)
    
def plot_error_by_hour_of_day(data, method, shift, forecast_horizon):
    data_h = pd.DataFrame(columns=['+'+str(i+1)+'h-prediction' for i in range(forecast_horizon)] + [method])
    for i in range(24):
        df = data.iloc[i::24]
        l = []
        for j in range(forecast_horizon):
            l.append(math.sqrt(mean_squared_error(df.measured, df['+'+str(j+1)+'h-prediction'])))
        l.append(math.sqrt(mean_squared_error(df.measured, df[method])))
        data_h.loc[i] = l

    for col in data_h.columns.values:
        data_h[col] = np.roll(data_h[col], shift)
        
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data_h.drop(['pvlib'], axis=1))
    ax.plot(data_h['pvlib'], color='black', linestyle='dashed', linewidth=2)
    ax.set_title('Mean RMSE per hour of day')
    ax.set_ylabel('RMSE [w]')
    ax.set_xlabel('Hour of day')
    ax.legend(data_h.columns.values)
    fig.autofmt_xdate()
    plt.grid(True)
    fig.savefig(dir + 'mean_RMSE_per_hour.png')
    plt.close(fig)
    
def daily_energy_error(m_col, p_col, l_col, method, horizon, start=None, end=None):
    j = int(len(m_col[start:end]) / 24)
    dm = np.array_split(m_col[start:end], j)
    dp = np.array_split(p_col[start:end], j)
    dl = np.array_split(l_col[start:end], j)
    
    energy_m = []
    energy_p = []
    energy_l = []
    abs_errors_p = []
    abs_errors_l = []
    errors_p = []
    errors_l = []
    for i in range(len(dm)):
        m = np.trapz(dm[i])
        p = np.trapz(dp[i])
        l = np.trapz(dl[i])
        abs_err_p = p - m
        abs_err_l = l - m
        abs_errors_p.append(abs_err_p)
        abs_errors_l.append(abs_err_l)
        errors_p.append(abs_err_p / m * 100)
        errors_l.append(abs_err_l / m * 100)
        energy_m.append(m)
        energy_p.append(p)
        energy_l.append(l)

    #print('daily mean ' + horizon + ' absolute energy error: ' + str(np.mean(abs_errors_p)))
    #print('daily mean ' + method + ' forecast absolute energy error: ' + str(np.mean(abs_errors_l)))
    #print('daily mean ' + horizon + ' relative energy error: ' + str(np.mean(errors_p)) + '%')
    #print('daily mean ' + method + ' forecast relative energy error: ' + str(np.mean(errors_l)) + '%')

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(errors_p, color='orange', linewidth=2)
    #ax.plot(errors_l, color='green', linestyle='dashed', linewidth=2)
    ax.set_title("Daily energy error")
    ax.set_ylabel("Energy error %")
    ax.set_yscale('linear')
    ax.legend([horizon, method])
    fig.autofmt_xdate()
    plt.grid(True)
    name = dir + horizon + '_rel_energy_error'
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    if l_col is not None:
        name += '_with_' + method
    fig.savefig(name + '.png')
    #fig.show()
    plt.close(fig)
    

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(abs_errors_p)
    ax.plot(abs_errors_l)
    ax.set_title("absolute daily energy error")
    ax.set_ylabel("Energy [Wh]")
    ax.set_yscale('linear')
    ax.legend([horizon, method])
    fig.autofmt_xdate()
    plt.grid(True)
    name = dir + horizon + '_abs_energy_error'
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    if l_col is not None:
        name += '_with_' + method
    fig.savefig(name + '.png')
    #fig.show()
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(energy_m)
    ax.plot(energy_p)
    #ax.plot(energy_l)
    ax.set_title("daily energy")
    ax.set_ylabel("Energy [Wh]")
    ax.set_yscale('linear')
    ax.legend(['measured', horizon])#, method])
    fig.autofmt_xdate()
    plt.grid(True)
    name = dir + horizon + '_daily_energy'
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    #if l_col is not None:
    #    name += '_with_' + method
    fig.savefig(name + '.png')
    #fig.show()
    plt.close(fig)
    
    
    fig, ax = plt.subplots(figsize=figsize)
    bp1 = _draw_boxplot(errors_p, 0, ax, 'black', 'orange', outliers=True)
    ax.set_title('Daily mean energy yield error')
    ax.set_ylabel('Error [%]')
    ax.set_yscale('log')
    ax.set_xticklabels([''])
    plt.grid(True)
    ax.legend([horizon + ' error'])

    name = dir + horizon + '_daily_mean_percentage_energy_error'
    if start:
        name += '_from' + start.replace(':', '-')
    if end:
        name += '_to' + end.replace(':', '-')
    fig.savefig(name + '.png', bbox_inches='tight')
    #fig.show()
    plt.close(fig)