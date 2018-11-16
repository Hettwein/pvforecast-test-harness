import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, History
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model
from keras.models import Model
from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

# 1.
# for sys in systems:
#   pre-train sys with pvlib forecasts
#   walk-forward validation for sys

# 2.
# for sys in systems:
#   pre-train sys with pvlib forecasts
# walking validation for all systems

# concepts:
#   -different preprocessing steps
#   -different input parameters
#   -different models
#   (-different hyper-parameters)
#   -different training forms
#   -evaluation


if True:
    # fix random seed for reproducibility
    np.random.seed(13)
        
    ## net params
    num_layers = 4
    num_neurons = 300#500
    batch_size = 100#500#1000
    dropout_rate = 0
    const_features = ['latitude', 'longitude', 'altitude', 'modules_per_string', 'strings_per_inverter', 'tilt',
                      'azimuth', 'albedo', 'Technology', 'BIPV', 'A_c', 'N_s', 'pdc0', 'gamma_pdc', 'SystemID']#15
    dyn_features = ['Wind Direction_x', 'Wind Direction_y', 'Total Cloud Cover', 'Low Cloud Cover', 'Medium Cloud Cover',
                    'High Cloud Cover', 'Wind Speed', 'Wind Gust', 'Total Precipitation',
                    'Snow Fraction', 'Mean Sea Level Pressure', 'DIF - backwards', 'DNI - backwards', 'Shortwave Radiation',
                    'Temperature', 'Relative Humidity', 'Hour_x', 'Hour_y', 'Month_x', 'Month_y']#20
    #const_features = ['SystemID']
    #dyn_features = ['DIF - backwards', 'DNI - backwards', 'Shortwave Radiation', 'Temperature', 'Relative Humidity', 'Hour_x', 'Hour_y', 'Month_x', 'Month_y']
    target_features = ['power']
    drop_features = ['power_pvlib']
    act_fct = 'relu'
    out_act = 'linear'
    loss_fct = 'mae'
    optim = 'adam'
    metrics = []
    history = History()
    val_history = History()

    ## data params
    filename = './data/full_data_5_systems.csv'
    correlations = ['pearson']#'pearson', 'spearman', 'kendall']
    timesteps = 5#24
    shape = (len(const_features) + len(dyn_features) + timesteps * (len(dyn_features) + len(target_features)),)
    forecast_horizon = 1

    ## training params
    tensorboard = False
    shuffle = True
    epochs = 20#100
    num_runs = 1
    val_split = 1.0/10.0
    dir = './test_results/'
    if not os.path.exists(dir):
        os.makedirs(dir)


def main():
    model = build_model()
    X, y, Xt, yt, idx, pvlib = prepare_data()
    for k in range(num_runs):
        print("Run "+str(k) + ":")
        #training(X, y, model)
        #evaluation(Xt, yt, idx, k, model, pvlib)
        randomForestRegression(X, y, Xt, yt, idx)

    
def build_model():
    print(shape)
    visible = Input(shape=shape)
    pre = visible
    for layer in range(0, num_layers):
        if dropout_rate > 0.0: 
            pre = Dropout(dropout_rate)(pre)
        pre = Dense(num_neurons, activation=act_fct)(pre)
    output = Dense(len(target_features), activation=out_act)(pre)
    model = Model(inputs=visible, outputs=output)
    model.compile(loss=loss_fct, optimizer=optim, metrics=metrics)
    plot_model(model, show_shapes=True, to_file=dir + 'model.png')
    return model


def prepare_data():
    pfname = dir + '/preprocessed_data_t-'+str(timesteps)+'_f'+str(shape[0])+'.csv'
    prep = Path(pfname)
    if prep.exists():
        print('Loading preprocessed dataset ...')
        pvlib = np.array_split(pd.read_csv(filename, skipinitialspace=True).set_index('time'), 5)[-1].power_pvlib
        dataset = pd.read_csv(pfname, skipinitialspace=True).set_index(['time', 'forecast_horizon', 'SystemID'])
    else:
        print('Data preprocessing ...')
        df = pd.read_csv(filename, skipinitialspace=True).set_index('time')
        df = np.array_split(df, 5)[-1]##################################
        pvlib = df.power_pvlib
        dataset = df[const_features + dyn_features + target_features].copy()[:'2017-02-09 10:00:00']
        
        #separate system
        for i in range(1, timesteps + 1):
            for feature in dyn_features + target_features:
                sys.stdout.write("Shifting %i/%i %s                \r" % (i, timesteps, feature))
                sys.stdout.flush()
                dataset[feature + ' t-' + str(i)] = dataset.shift(i)[feature]
        print('Shifting done.                ')
        
        data = pd.DataFrame()
        for i in range(forecast_horizon):
            sys.stdout.write("Adding forecast horizon %i/%i                \r" % (i+1, forecast_horizon))
            sys.stdout.flush()
            d = dataset.copy()
            n = len(const_features) + len(const_features) + len(target_features) + 1
            d.iloc[:,n:] = d.iloc[:,n:].shift(i)
            d['forecast_horizon'] = i
            d['horizon'] = i
            data = data.append(d)
        print('Horizons done.                ')
        
        p = data[target_features]
        data = data.drop(target_features, axis=1)
        for f in target_features:
            data[f] = p[f]
        dataset = data[forecast_horizon:].dropna().reset_index().set_index(['time', 'forecast_horizon', 'SystemID'])
        
        sys.stdout.write("Writing to file ...\r")
        sys.stdout.flush()
        dataset.to_csv(pfname, encoding='utf-8')
        print('Writing done.                ')
    
        if correlations:
            sys.stdout.write('Computing correlations ...\r')
            sys.stdout.flush()
            for corr in correlations:
                sys.stdout.write("Computing %s correlation matrix                \r" % (corr))
                sys.stdout.flush()
                dataset.corr(method=corr).to_csv(dir + corr + '_correlations.csv', encoding='utf-8')
            print('Correlations done.                   ')
    
    train, test = dataset[:('2015-10-12 06:00:00', 0, 4.0)], dataset[('2015-10-12 07:00:00', 0, 4.0):]
    trainX, trainY = train.iloc[:,:-len(target_features)], train.iloc[:,-len(target_features):]
    testX, testY = test.iloc[:,:-len(target_features)], test.iloc[:,-len(target_features):]

    return trainX, trainY, testX, testY, testX.index.values, pvlib

def randomForestRegression(x_train, y_train, x_test, y_test, dates):
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(n_estimators=216, criterion='mse', oob_score=True, n_jobs=4, verbose=1, max_features=0.53)#, random_state=0
    regr.fit(x_test, y_test)#x_train, y_train)
    
    features = x_train.columns
    importances = regr.feature_importances_
    indices = np.argsort(importances)[-29:]  # top 30 features
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
    
    train_pred = regr.predict(x_train)
    test_pred = regr.predict(x_test)
    mse_train = mean_squared_error(y_train, train_pred)
    mse_test = mean_squared_error(y_test, test_pred)

    print("RMSE Training: %f" % math.sqrt(mse_train))
    print("RMSE Test: %f" % math.sqrt(mse_test))

    #dates = pd.to_datetime(dates)
    y_test = pd.Series(y_test)#, index=dates)
    test_pred = pd.Series(test_pred)#, index=dates)
    
    plt.figure()
    plt.plot(y_test, label='Test')
    plt.plot(test_pred, label='RandomForest')#, color='red') #, index=y_test.index
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.plot(y_test, label='Test')
    plt.plot(test_pred, label='RandomForest')#, color='red') #, index=y_test.index
    plt.legend(loc='best')
    plt.show()
    

def training(features, labels, model):
    if shuffle:
        df = pd.DataFrame(np.concatenate((features, labels), axis=1))
        df = df.sample(frac=1).values
        labels = df[:, -len(target_features):]
        features = df[:, :-len(target_features)]

    if tensorboard:
        print('tensorboard activated')
        callbacks = [TensorBoard(log_dir='./tensorboard', histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=True, write_images=False), history]
    else:
        callbacks = [history]
        
    model.fit(features, labels, batch_size, epochs=epochs, validation_split=val_split, callbacks=callbacks, verbose=1)


def _draw_boxplot(data, offset, ax, edge_color, fill_color, num=1):
    pos = np.arange(num) + offset 
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, whis='range', manage_xticks=False)
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
    plt.close(fig)
    
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
    bp1 = _draw_boxplot(preds, -0.2, ax, 'red', 'tomato', len(preds))
    bp2 = _draw_boxplot(pvlibs, 0.2, ax, 'blue', 'cornflowerblue', len(pvlibs))
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], [horizon, method])
    ax.set_xticklabels(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', 'january'])
    fig.savefig(dir + 'boxplot_' + horizon + '_vs_' + method + '_monthly.png', bbox_inches='tight')
    plt.close(fig)

def walkForwardDailyLoss(test_y, pred_y, method_y, method, horizon):
    #name = '+' + str(horizon) + 'h-prediction'
    #df = data#.replace(0, np.nan).dropna()
    j = int(len(test_y) / 24)
    d1 = np.array_split(test_y, j)
    d2 = np.array_split(pred_y, j)
    d3 = np.array_split(method_y, j)
    pred_error = pd.DataFrame([math.sqrt(mean_squared_error(d2[i], d1[i])) for i in range(len(d1))])
    method_error = pd.DataFrame([math.sqrt(mean_squared_error(d3[i], d1[i])) for i in range(len(d1))])
    print('mean ' + horizon + ' RMSE: ' + str(pred_error.mean()[0]))
    print('mean ' + method + ' forecast RMSE: ' + str(method_error.mean()[0]))
    
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
    plt.close(fig)

def scatter_predictions(test_y, pred_y, horizon):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(test_y, pred_y)
    ax.plot([np.amin(test_y), np.amax(test_y)], [np.amin(pred_y), np.amax(pred_y)], 'k--', lw=3)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    fig.savefig(dir + 'scatter_' + horizon + '.png')
    plt.close(fig)
    
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
    plt.close(fig)

def draw_histogram(p_col, m_col, horizon):
    fig, ax = plt.subplots(figsize=(12, 12)) 
    ax.hist(p_col - m_col, bins=200)
    fig.savefig(dir + horizon + '_histogram.png')
    plt.close(fig)
    
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
    plt.close(fig)
    
def evaluation(test_features, test_labels, dates, k, model, pvlib):
    method = 'pvlib'
    
    predictions = []
    for i in range(len(test_features)):
        sys.stdout.write("Walk-Forward Validation %i/%i\r" % (i, len(test_features)))
        sys.stdout.flush()
        predictions.append(pd.DataFrame(model.predict(test_features.iloc[i:i+1,:], 1, verbose=0)))
        #batch_size = 2
        #dfx = test_features.iloc[i:i+1,:]
        #dfx = dfx.append([dfx]*(batch_size-1), ignore_index=True)
        #dfy = pd.DataFrame(test_labels.iloc[i])
        #dfy = dfy.append([dfy]*(batch_size-1), ignore_index=True)
        #test with window!!, test with t-24, ...
        window = i - 24
        if window < 0:
            window = 0
        model.fit(test_features.iloc[window:i+1,:], test_labels.iloc[window:i+1,:], epochs=epochs, verbose=0, callbacks=[val_history])
    prediction = pd.concat(predictions)

    data = pd.DataFrame()
    data['prediction'] = pd.DataFrame(np.array(prediction).reshape([len(prediction), len(target_features)])).iloc[:,0]
    data['measured'] = pd.DataFrame(np.array(test_labels).reshape([len(test_labels), len(target_features)])).iloc[:,0]
    data = data.set_index(pd.MultiIndex.from_tuples(dates)).unstack().unstack()
    #data = data.drop(data.columns.values[forecast_horizon+1:], axis=1)
    data['pvlib'] = pvlib['2015-10-12 07:00:00':'2017-02-09 10:00:00'].reindex(data.index)
    
    tmp = pd.DataFrame()
    for i in range(forecast_horizon):
        tmp['+' + str(i+1) + 'h-prediction'] = data[('prediction', 4.0, i)]
    tmp[method] = data[method]
    tmp['measured'] = data[('measured', 4.0, 0)]
    data = tmp
    data.index = pd.to_datetime(data.index)
    
    
    m_col = data['measured']
    l_col = data[method].dropna()

    for horizon in range(1, forecast_horizon + 1):
        name = '+' + str(horizon) + 'h-prediction'
        p_col = data[name]

        walkForwardDailyLoss(m_col, p_col, l_col, method, name)
        scatter_predictions(m_col, p_col, name)

        print('%s test RMSE: %.3f' % (name, math.sqrt(mean_squared_error(m_col, p_col))))
        print('%s test RMSE: %.3f' % (method + ' forecast', math.sqrt(mean_squared_error(m_col, l_col))))
        draw_boxplot(m_col, p_col, l_col, method, name)
        draw_boxplot_monthly(m_col, p_col, l_col, method, name)
        
        m1, m2 = '2016-07-17 00:00:00', '2016-07-17 23:00:00'
        print('%s nice day RMSE: %.3f' % (name, math.sqrt(mean_squared_error(m_col[m1:m2], p_col[m1:m2]))))
        print('%s nice day RMSE: %.3f' % (method + ' forecast', math.sqrt(mean_squared_error(m_col[m1:m2], l_col[m1:m2]))))
        draw_boxplot(m_col, p_col, l_col, method, name, m1, m2)

        plot_timeseries(m_col, p_col, l_col, method, name, end='2015-10-19 07:00:00')
        plot_timeseries(m_col, p_col, l_col, method, name, start='2017-02-02 10:00:00')
        plot_timeseries(m_col, p_col, l_col, method, name, start=m1, end=m2)
        plot_timeseries(m_col, p_col, l_col, method, name)
        plot_timeseries(m_col, p_col, None, method, name)
        
        draw_histogram(p_col, m_col, name)
        
    draw_history(history)
    draw_history(val_history, True)
               
    print(data.describe())
    print(data.corr(method='pearson'))
    print(data.corr(method='spearman'))
    print(data.corr(method='kendall'))
    data.to_csv(dir + 'predictions' + str(k) + '.csv', encoding='utf-8')
    

if __name__ == "__main__":
    main()