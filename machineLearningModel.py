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
            pre = Dense(num_neurons, activation=act_fct)(pre)
        output = Dense(output_size)(pre)
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
    

    
class ConvolutionalNeuralNetwork(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('ConvolutionalNeuralNetwork')
        
        visible = Input(shape=input_shape)
        pre = Conv1D(filters=64, kernel_size=2, activation=act_fct)(visible)
        pre = MaxPooling1D(pool_size=2)(pre)
        pre = Flatten()(pre)
        pre = Dense(num_neurons, activation=act_fct)(pre)
        output = Dense(output_size)(pre)
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
    
    def learn(self, X, y):
        return self.model.fit(X, y, self.batch_size, self.epochs, self.verbose, self.callbacks,
                              self.validation_split, shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)
    

    
class LongShortTermMemory(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('LongShortTermMemory')
        
        model = Sequential()
        if num_layers > 1:
            model.add(LSTM(num_neurons, activation=act_fct, return_sequences=True, input_shape=input_shape))
            for layer in range(1, num_layers-1):
                model.add(LSTM(num_neurons, activation=act_fct, return_sequences=True))
            model.add(LSTM(num_neurons, activation=act_fct))
        else:
            model.add(LSTM(num_neurons, activation=act_fct, input_shape=input_shape))
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
    
    def __init__(self, input_shape, output_size, num_layers, ks=3, fs=32, pa='causal', loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('DilatedConvolution')
        
        model = Sequential()
        model.add(Conv1D(fs, ks, input_shape=input_shape, activation=act_fct, dilation_rate=1, padding="causal"))
        for n in range(num_layers):
            model.add(Conv1D(fs, ks, activation=act_fct, dilation_rate=2**(n+1), padding=pa))
        model.add(Flatten())
        model.add(Dense(50, activation=act_fct))
        model.add(Dense(50, activation=act_fct))
        model.add(Dense(50, activation=act_fct))
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
    

        
class CNNLSTM(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('CNNLSTM')
        
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation=act_fct), input_shape=(None, input_shape[0], input_shape[1])))#(None, n_steps, n_features))) #input_shape
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(num_neurons, activation=act_fct))
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
    
    def learn(self, X, y):
        return self.model.fit(X, y, self.batch_size, self.epochs, self.verbose, self.callbacks,
                              self.validation_split, shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)

    
    
class ConvLSTM(MachineLearnModel):
    
    def __init__(self, input_shape, output_size, num_layers, num_neurons, loss_fct='mse', optim='adam',
                 act_fct='relu', out_act='linear', metrics=[], dropout_rate=0, plot=None, batch_size=1,
                 epochs=1, validation_split=0.0, callbacks=None, verbose=1, shuffle=True, **kwargs):
        super().__init__('ConvLSTM')
        
        model = Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation=act_fct, input_shape=(1, 1, input_shape[0], input_shape[1])))#(n_seq, 1, n_steps, n_features))) #input_shape
        model.add(Flatten())
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
    
    def learn(self, X, y):
        return self.model.fit(X, y, self.batch_size, self.epochs, self.verbose, self.callbacks,
                              self.validation_split, shuffle=self.shuffle)
    
    def forecast(self, X):
        return self.model.predict(X, self.batch_size, self.verbose)  
    
    
    
class RandomForest(MachineLearnModel):
    
    def __init__(self, est, mf, crit='mse', **kwargs):
        super().__init__('RandomForest')
        self.model = RandomForestRegressor(n_estimators=est, criterion=crit, oob_score=False, n_jobs=4, verbose=0, max_features=mf)#True
    
    def learn(self, X, y, **kwargs):
        return self.model.fit(X.values, y.values.ravel())
    
    def forecast(self, X, **kwargs):
        return self.model.predict(X.values)
        