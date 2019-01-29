from keras.layers import Input, Dense, Dropout, Flatten, LSTM, ConvLSTM2D, TimeDistributed, LeakyReLU, ReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model, Sequential
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from abc import ABC, abstractmethod
from keras.regularizers import l1, l2, l1_l2
from sklearn.ensemble import RandomForestRegressor



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
            pre = Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform')(pre)
        output = Dense(output_size, kernel_initializer='he_uniform')(pre)
        output = LeakyReLU(alpha=0.001)(output)

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
    
    def learn(self, X, y, val_idx=0):
        return self.model.fit(X[val_idx:], y[val_idx:], self.batch_size, self.epochs, self.verbose, self.callbacks,
                              validation_data=(X[:val_idx], y[:val_idx]), shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)
    

    
class LongShortTermMemory(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('LongShortTermMemory')
        
        model = Sequential()
        if num_layers > 1:
            model.add(LSTM(num_neurons, activation=act_fct, return_sequences=True, input_shape=input_shape, kernel_initializer='he_uniform'))
            for layer in range(1, num_layers-1):
                model.add(LSTM(num_neurons, activation=act_fct, return_sequences=True, kernel_initializer='he_uniform'))
            model.add(LSTM(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        else:
            model.add(LSTM(num_neurons, activation=act_fct, input_shape=input_shape, kernel_initializer='he_uniform'))
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(output_size))
        model.add(LeakyReLU(alpha=0.001))
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
    
    def learn(self, X, y, val_idx=0):
        return self.model.fit(X[val_idx:], y[val_idx:], self.batch_size, self.epochs, self.verbose, self.callbacks,
                              validation_data=(X[:val_idx], y[:val_idx]), shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)
    

    
class DilatedConvolution(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, ks=3, fs=32, pa='causal', loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('DilatedConvolution')
        
        model = Sequential()
        model.add(Conv1D(fs, ks, input_shape=input_shape, activation=act_fct, dilation_rate=1, padding="causal", kernel_initializer='he_uniform'))
        for n in range(num_layers):
            model.add(Conv1D(fs, ks, activation=act_fct, dilation_rate=2**(n+1), padding=pa, kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(num_neurons, activation=act_fct, kernel_initializer='he_uniform'))
        model.add(Dense(output_size))
        model.add(LeakyReLU(alpha=0.001))
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
    
    def learn(self, X, y, val_idx=0):
        return self.model.fit(X[val_idx:], y[val_idx:], self.batch_size, self.epochs, self.verbose, self.callbacks,
                              validation_data=(X[:val_idx], y[:val_idx]), shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)
    
   
    
class RandomForest(MachineLearnModel):
    
    def __init__(self, est, mf, crit='mse', verbose=0, **kwargs):
        super().__init__('RandomForest')
        self.model = RandomForestRegressor(n_estimators=est, criterion=crit, n_jobs=-1, verbose=verbose, max_features=mf, random_state=1)#, max_depth=2 min_samples_leaf=0.05
    
    def learn(self, X, y, **kwargs):
        return self.model.fit(X, y.ravel())
    
    def forecast(self, X, **kwargs):
        return self.model.predict(X)
        