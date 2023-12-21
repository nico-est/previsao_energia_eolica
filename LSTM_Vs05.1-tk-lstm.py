from sklearn.preprocessing import MinMaxScaler

#from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model 
from tensorflow.keras.callbacks import ModelCheckpoint



from tensorflow.keras.utils import plot_model

from tensorflow.keras.optimizers import SGD

#Versão Nova TensorFlow
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM, Dropout
#from keras.layers import LayerNormalization
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


import json

from keras_tuner.tuners import BayesianOptimization


import os

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.force_gpu_compatible=True
sess = tf.compat.v1.Session(config=config)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print('Versão TensorFlow:' + tf.__version__)

print('Versão NumPy:' + np.__version__)


#DataSet Train 2015 - 2017
Data_Ini_Train='01-01-2015 01:00:00'
Data_Fim_Train='31-12-2017 23:00:00'

Data_Ini_Train='2015-01-01 01:00:00'
Data_Fim_Train='2017-12-31 23:00:00'

#DataSet Validação 2018
Data_Ini_val='2018-01-01 01:00:00'
Data_Fim_val='2018-12-31 23:00:00'

#DataSet Teste 2019 - 2020
Data_Ini_Test='2019-01-01 01:00:00'
Data_Fim_Test='2020-12-31 23:00:00'


arq=os.getcwd()+'\\Setembro'+'\\pares_guanambi_normalizados.csv'
dados_normalizados = pd.read_csv(arq, engine='python',sep=',',encoding='latin-1')


arq=os.getcwd()+'\\Setembro'+'\\pares_guanambi.csv'
dados = pd.read_csv(arq, engine='python',sep=',',encoding='latin-1').set_index('Tempo')






np.random.seed(7)

#Conjutno de Treino(2015 - 2017) e Validação(2018)

VDatas=dados_normalizados['Tempo']  
pos_train1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_Train].index.values[0]
pos_train2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_Train].index.values[0]

pos_val1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_val].index.values[0]
pos_val2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_val].index.values[0]


#Reespecifica o dataset pra ficar no horizonte dos conjuntos de Treinamento e Validação. Pois é usado mais abaixo no código
dataset_tv = dados.reset_index()[pos_train1:pos_val2]


    
VDatas_test = dados_normalizados['Tempo']

pos_test1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_Test].index.values[0]
pos_test2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_Test].index.values[0]



dataset_test = dados[pos_test1:pos_test2]




def create_dataset(dataset, look_back, look_forward, cols_del=[]):
    dataX, dataY = [], []
    dataset = dataset.drop(cols_del, axis=1).values  # Drop unnecessary columns

    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back), :]  # Use .iloc to slice the DataFrame
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back + look_forward, 0])  # Assuming 'Power' is in the first column

    return np.array(dataX), np.array(dataY)



look_back = 24
look_forward = 12
cols_del = []

usinas =  ['geração usina alvorada', 'geração usina aracas',
       'geração usina caetite_123', 'geração usina guirapa',
       'geração usina licinio_almeida', 'geração usina morrao',
       'geração usina planaltina']


for i in usinas:

    
    
    train, val = dataset_tv[pos_train1:pos_train2], dataset_tv[pos_val1:pos_val2]
    test = dados_normalizados[pos_test1:pos_test2]
    
    
    columns_to_keep = ['Temperatura', 'Velocidade do Vento', 'Direção do Vento', i]
    train = train[columns_to_keep]
    train_power = train.pop(i)
    train.insert(0, i, train_power)
    
    val = val[columns_to_keep]
    val_power = val.pop(i)
    val.insert(0, i, val_power)
    
    test = test[columns_to_keep]
    test_power = test.pop(i)
    test.insert(0, i, test_power)
    
    trainX, trainY = create_dataset(train, look_back, look_forward, cols_del)
    valX, valY = create_dataset(val, look_back, look_forward, cols_del)
    testX, testY = create_dataset(test, look_back, look_forward, cols_del)  
    
    # Reshape para 2D
    trainX = trainX.reshape(-1, trainX.shape[2])
    valX = valX.reshape(-1, valX.shape[2])
    testX = testX.reshape(-1, testX.shape[2])
    
    
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize the training data
    normalized_trainX = scaler.fit_transform(trainX)
    
    # Normalize the validation data
    normalized_valX = scaler.transform(valX)
    
    # Normalize the test data
    normalized_testX = scaler.transform(testX)
        
    # Reshape the data to 3D
    normalized_trainX = normalized_trainX.reshape(normalized_trainX.shape[0], 1, normalized_trainX.shape[1])
    normalized_valX = normalized_valX.reshape(normalized_valX.shape[0], 1, normalized_valX.shape[1])
    normalized_testX = normalized_testX.reshape(normalized_testX.shape[0], 1, normalized_testX.shape[1])

    
    es = EarlyStopping(
        monitor='val_loss', 
        patience=30, 
        min_delta=0.001, 
        mode='min'             
    )       
     
    mc = ModelCheckpoint('Plot/Mult/Usina-'+ i + '-best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
        
    param = 1e-3 # values
    
    batch_size = [1]
    
    all_train, all_test, all_val = list(), list(), list()
    nmape_train, nmape_test, nmape_val = list(), list(), list()
    
    
    with tf.device('/device:GPU:0'):
        
        NUM_EPOCHS = 5
        
        if True:
        
            opt = SGD(lr=0.1, momentum=0.0, decay=1E-4)
            
                   
            b_model = Sequential()
            b_model.add(LSTM(1024, kernel_regularizer=l2(float(param)), kernel_initializer='he_normal', return_sequences = False, input_shape=(trainX.shape[1],)))
            # b_model.add(Dropout(0.3))
            b_model.add(Dense(512, kernel_regularizer=l2(float(param)), activation='relu'))            
            b_model.add(Dropout(0.4))  
            b_model.add(Dense(look_forward))
     
           
            b_model.compile(loss='mean_squared_error', optimizer=opt , metrics=['mse', tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.MeanAbsoluteError()])
            history = b_model.fit(normalized_trainX, trainY, validation_data=(normalized_valX,valY), epochs=NUM_EPOCHS, batch_size=32, verbose=1, callbacks=[es,mc]) #inserir mc
            
            #saved_model = load_model('Plot/Mult/Usina-alvorada-best_model.h5') 
            b_model.save('ultimo_modelo_' + i + '.h5')    
            
            plot_model(b_model, to_file='Plot/Mult/Usina-'+ i + '-lstm-b_model.png', show_shapes=True, show_layer_names=True)
            
            
            train_loss, train_mse, train_mape, train_mae = b_model.evaluate(normalized_trainX, trainY, verbose=1)
            val_loss, val_mse, val_mape, val_mae = b_model.evaluate(normalized_testX, testY, verbose=1)
            test_loss, test_mse, test_mape, test_mae = b_model.evaluate(normalized_valX, valY, verbose=1)
            
            print('LOSS Train: %.4f, Val: %.4f, Test: %.4f' % (train_loss, val_loss, test_loss))
            
            print('MSE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mse, val_mse, test_mse))
            
            print('MAPE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mape, val_mape, test_mape))
            
            print('MAE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mae, val_mae, test_mae))
            
            
           
     
            b_model.summary()  
        
    
    trainPredict = b_model.predict(normalized_trainX)
    testPredict = b_model.predict(normalized_testX)
    valPredict = b_model.predict(normalized_valX)
    
    
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    valPredict = scaler.inverse_transform(valPredict)
    
    #trainPredict = scaler_pred.inverse_transform(trainPredict)
    #testPredict = scaler_pred.inverse_transform(testPredict)
    #valPredict = scaler_pred.inverse_transform(valPredict)
    
    trainPredict = trainPredict.reshape((-1, 1))
    testPredict = testPredict.reshape((-1, 1))
    valPredict = valPredict.reshape((-1, 1))
    
    #trainY = scaler_pred.inverse_transform(trainY)
    #testY = scaler_pred.inverse_transform(testY)
    #valY = scaler_pred.inverse_transform(valY)

    
    
    trainY = trainY.reshape((-1, 1))
    testY = testY.reshape((-1, 1))
    valY = valY.reshape((-1, 1))
    
    
    X_train = np.arange(len(trainY))
    X_val = np.arange(len(valY))
    X_test = np.arange(len(testY))
    
    
    plt.plot(X_train, trainY, label = 'Real', linewidth = 0.5)
    plt.plot(X_train, trainPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Treino')
    plt.legend()
    plt.savefig('Plot/treino-' + i + '.pdf', bbox_inches='tight')
    plt.show()
    
    
    plt.plot(X_val, valY, label = 'Real', linewidth = 0.5)
    plt.plot(X_val, valPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Validação')
    plt.legend()
    plt.savefig('Plot/val-' + i + '.pdf')
    plt.show()
    
    plt.plot(X_test, testY, label = 'Real', linewidth = 0.5)
    plt.plot(X_test, testPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Teste')
    plt.savefig('Plot/teste-' + i + '.pdf', bbox_inches='tight')
    plt.legend()
    plt.show()
    
    
    

    
    
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-loss-'+ i + '.pdf', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mse-'+ i + '.pdf', bbox_inches='tight')
    plt.show()


    print(history.history.keys())
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('mean_absolute_percentage_error')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mape-'+ i + '.pdf', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mae-'+ i + '.pdf', bbox_inches='tight')
    plt.show()





