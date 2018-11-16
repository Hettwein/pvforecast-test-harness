from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model
from keras.models import Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from abc import ABC, abstractmethod
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX



class MachineLearnModel(ABC):
    
    def __init__(self, name):
        print('Using ' + name + '\n')
                 
    @abstractmethod
    def learn(self, X, y, **kwargs):
        pass
    
    @abstractmethod
    def forecast(self, X, **kwargs):
        pass
    
    
    
class MultiLayerPerceptron(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('MultiLayerPerceptron')
        visible = Input(shape=input_shape)
        pre = visible
        for layer in range(0, num_layers):
            if dropout_rate > 0.0: 
                pre = Dropout(dropout_rate)(pre)
            pre = Dense(num_neurons, activation=act_fct)(pre)
        output = Dense(output_size, activation=out_act)(pre)
        model = Model(inputs=visible, outputs=output)
        model.compile(loss=loss_fct, optimizer=optim, metrics=metrics)
        if plot:
            plot_model(model, show_shapes=True, to_file=plot)
            SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.verbose = verbose
        self.shuffle = shuffle
    
    def learn(self, X, y):
        return self.model.fit(X, y, self.batch_size, self.epochs, self.verbose, self.callbacks,
                              self.validation_split, shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)
    

    
class ARIMA(MachineLearnModel):
    
    def __init__(self, order=(1, 0, 0), **kwargs):
        super().__init__('ARIMA')
        self.order = order
    
    def learn(self, X, y):
        model = ARIMA(y, order=self.order)#exog=X
        self.model = model.fit()
        return self.model
    
    def forecast(self, X):
        return self.model.forecast()#predict()#exog=X)
    


class SARIMAX(MachineLearnModel):
    
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), **kwargs):
        super().__init__('SARIMA')
        self.order = order
        self.sorder = seasonal_order
    
    def learn(self, X, y):
        model = SARIMAX(y, exog=X, order=self.order, seasonal_order=self.sorder, enforce_stationarity=True, enforce_invertibility=True)
        self.model = model.fit()
        return self.model
    
    def forecast(self, X):
        return self.model.predict(exog=X)