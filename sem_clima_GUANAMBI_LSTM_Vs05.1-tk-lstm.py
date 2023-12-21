from sklearn.preprocessing import MinMaxScaler

#from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model 
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime, timedelta



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
import tensorflow.keras.metrics
from tensorflow_addons.metrics import RSquare

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

Data_Ini_Train='2015-01-01 01:00:00'
Data_Fim_Train='2017-12-31 23:00:00'

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



scaler_temp = MinMaxScaler()
scaler_vel = MinMaxScaler()
scaler_dir = MinMaxScaler()
scaler_alvorada = MinMaxScaler()
scaler_aracas = MinMaxScaler()
scaler_caetite = MinMaxScaler()
scaler_guirapa = MinMaxScaler()
scaler_licinio = MinMaxScaler()
scaler_morrao = MinMaxScaler()
scaler_planaltina = MinMaxScaler()

dados['Temperatura'] = scaler_temp.fit_transform(dados['Temperatura'].values.reshape(-1, 1))
dados['Velocidade do Vento'] = scaler_vel.fit_transform(dados['Velocidade do Vento'].values.reshape(-1, 1))
dados['Direção do Vento'] = scaler_dir.fit_transform(dados['Direção do Vento'].values.reshape(-1, 1))

dados['geração usina alvorada'] = scaler_alvorada.fit_transform(dados['geração usina alvorada'].values.reshape(-1, 1))

dados['geração usina aracas'] = scaler_aracas.fit_transform(dados['geração usina aracas'].values.reshape(-1, 1))

dados['geração usina caetite_123'] = scaler_caetite.fit_transform(dados['geração usina caetite_123'].values.reshape(-1, 1))

dados['geração usina guirapa'] = scaler_guirapa.fit_transform(dados['geração usina guirapa'].values.reshape(-1, 1))

dados['geração usina licinio_almeida'] = scaler_licinio.fit_transform(dados['geração usina licinio_almeida'].values.reshape(-1, 1))

dados['geração usina morrao'] = scaler_morrao.fit_transform(dados['geração usina morrao'].values.reshape(-1, 1))

dados['geração usina planaltina'] = scaler_planaltina.fit_transform(dados['geração usina planaltina'].values.reshape(-1, 1))


scalers = [scaler_alvorada, scaler_aracas, scaler_caetite, scaler_guirapa, scaler_licinio, scaler_morrao, scaler_planaltina]

np.random.seed(7)

#Conjutno de Treino(2015 - 2017) e Validação(2018)

VDatas=dados_normalizados['Tempo']  
pos_train1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_Train].index.values[0]
pos_train2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_Train].index.values[0]

pos_val1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_val].index.values[0]
pos_val2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_val].index.values[0]


#Reespecifica o dataset pra ficar no horizonte dos conjuntos de Treinamento e Validação. Pois é usado mais abaixo no código
dataset_tv = dados.reset_index()[pos_train1:pos_val2]
train, val = dataset_tv[pos_train1:pos_train2], dataset_tv[pos_val1:pos_val2]


    
VDatas_test = dados_normalizados['Tempo']

pos_test1=dados_normalizados[dados_normalizados['Tempo'] == Data_Ini_Test].index.values[0]
pos_test2=dados_normalizados[dados_normalizados['Tempo'] == Data_Fim_Test].index.values[0]


dataset_test = dados[pos_test1:pos_test2]
test = dados[pos_test1:pos_test2]



def create_dataset(dataset, look_back, look_forward, cols_del=[]):
    dataX, dataY = [], []
    dataset = dataset.drop(cols_del, axis=1).values  # Drop unnecessary columns

    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back), :]  # Use .iloc to slice the DataFrame
        dataX.append(a)
        dataY.append(dataset[i + look_back:i + look_back + look_forward, 0])  # Assuming 'Power' is in the first column

    return np.array(dataX), np.array(dataY)

def average(lst):
    return sum(lst)/len(lst)


look_back = 24
look_forward = 12
cols_del = []

usinas =  ['geração usina alvorada', 'geração usina aracas',
       'geração usina caetite_123', 'geração usina guirapa',
       'geração usina licinio_almeida', 'geração usina morrao',
       'geração usina planaltina']


usinas = ['geração usina morrao', 'geração usina aracas']

cont = 0
for i in usinas:

    
    #normalized_df = (dados - dados.min()) / (dados.max() - dados.min())
 
    
    #normalized_df = (dados - dados.min()) / (dados.max() - dados.min())
    #normalized_df['Power'] = scaler_pred.fit_transform(dados[i].values.reshape(-1, 1))
    
    train, val = dataset_tv[pos_train1:pos_train2], dataset_tv[pos_val1:pos_val2]
    test = dados[pos_test1:pos_test2]
    
    
    columns_to_keep = ['Temperatura', 'Velocidade do Vento', 'Direção do Vento', i]
    columns_to_keep = [i]
    
    train = train[columns_to_keep]
    # train_power = train.pop(i)
    # train.insert(0, i, train_power)
    
    val = val[columns_to_keep]
    # val_power = val.pop(i)
    # val.insert(0, i, val_power)
    
    test = test[columns_to_keep]
    # test_power = test.pop(i)
    # test.insert(0, i, test_power)
    
    trainX, trainY = create_dataset(train, look_back, look_forward, cols_del)
    valX, valY = create_dataset(val, look_back, look_forward, cols_del)
    testX, testY = create_dataset(test, look_back, look_forward, cols_del)  
    
    
    
    es = EarlyStopping(
        monitor='val_loss', 
        patience=30, 
        min_delta=0.001, 
        mode='min'             
    )       
    
    ##################################################################################
   
    
    mc = ModelCheckpoint('Plot/Mult/Usina-'+ i + '-best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
        
    param = 1e-3 # values
    
    
    loss_train, loss_val, loss_test = list(), list(), list()
    mse_train, mse_val, mse_test = list(), list(), list()
    mape_train, mape_val, mape_test = list(), list(), list()
    mae_train, mae_val, mae_test = list(), list(), list()
    r2_train, r2_val, r2_test = list(), list(), list()
    rmse_train, rmse_val, rmse_test = list(), list(), list()
    

    
    
    with tf.device('/device:GPU:0'):
        
        NUM_EPOCHS = 30
        
        if True:
        
            opt = SGD(lr=0.1, momentum=0.0, decay=1E-4)
            
                   
            b_model = Sequential()
            b_model.add(LSTM(1024, kernel_regularizer=l2(float(param)), kernel_initializer='he_normal', return_sequences = False, input_shape=(trainX.shape[1], trainX.shape[2])))
            # b_model.add(Dropout(0.3))
            b_model.add(Dense(512, kernel_regularizer=l2(float(param)), activation='relu'))            
            b_model.add(Dropout(0.4))  
            b_model.add(Dense(look_forward))
     
           
            b_model.compile(loss='mean_squared_error', optimizer=opt , 
                            metrics=['mse',
                                     tf.keras.metrics.MeanAbsolutePercentageError(),
                                     tf.keras.metrics.MeanAbsoluteError(),
                                     RSquare(),
                                     tf.keras.metrics.RootMeanSquaredError()])
           
            
            history = b_model.fit(trainX, trainY, validation_data=(valX,valY), epochs=NUM_EPOCHS, batch_size=32, verbose=1, callbacks=[es,mc]) #inserir mc
            
            #saved_model = load_model('Plot/Mult/Usina-alvorada-best_model.h5') 
            b_model.save('ultimo_modelo_' + i + '.h5')    
            
            plot_model(b_model, to_file='Plot/Mult/Usina-'+ i + '-lstm-b_model.png', show_shapes=True, show_layer_names=True)
            
            
            train_loss, train_mse, train_mape, train_mae, train_r2, train_rmse = b_model.evaluate(trainX, trainY, verbose=1)
            val_loss, val_mse, val_mape, val_mae, val_r2, val_rmse = b_model.evaluate(valX, valY, verbose=1)
            test_loss, test_mse, test_mape, test_mae, test_r2, test_rmse = b_model.evaluate(testX, testY, verbose=1)
            
            print('LOSS Train: %.4f, Val: %.4f, Test: %.4f' % (train_loss, val_loss, test_loss))
            
            print('MSE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mse, val_mse, test_mse))
            
            print('MAPE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mape, val_mape, test_mape))
            
            print('MAE Train: %.4f, Val: %.4f, Test: %.4f' % (train_mae, val_mae, test_mae))
            
            print('R2 Train: %.4f, Val: %.4f, Test: %.4f' % (train_r2, val_r2, test_r2))
            
            print('RMSE Train: %.4f, Val: %.4f, Test: %.4f' % (train_rmse, val_rmse, test_rmse))
            
            loss_train.append(train_loss)
            loss_val.append(val_loss)
            loss_test.append(test_loss)
            
            mse_train.append(train_mse)
            mse_val.append(val_mse)
            mse_test.append(test_mse)
            
            mape_train.append(train_mape)
            mape_val.append(val_mape)
            mape_test.append(test_mape)
            
            mae_train.append(train_mae)
            mae_val.append(val_mae)
            mae_test.append(test_mae)
            
            r2_train.append(train_r2)
            r2_val.append(val_r2)
            r2_test.append(test_r2)
            
            rmse_train.append(train_rmse)
            rmse_val.append(val_rmse)
            rmse_test.append(test_rmse)
            
            
           
     
            b_model.summary()  
     
        

    with open('output_' + i +'_sem_dados_climaticos.txt', 'w') as f:
        f.write('Average LOSS: Train %.4f, Val %.4f, Test %.4f' % (average(loss_train), average(loss_val), average(loss_test)))
        f.write('Average MSE: Train %.4f, Val %.4f, Test %.4f \n' % (average(mse_train), average(mse_val), average(mse_test)))
        f.write('Average MAPE: Train %.4f, Val %.4f, Test %.4f\n' % (average(mape_train), average(mape_val), average(mape_test)))
        f.write('Average MAE: Train %.4f, Val %.4f, Test %.4f\n' % (average(mae_train), average(mae_val), average(mae_test)))
        f.write('Average R2: Train %.4f, Val %.4f, Test %.4f\n' % (average(r2_train), average(r2_val), average(r2_test)))
        f.write('Average RMSE: Train %.4f, Val %.4f, Test %.4f\n'% (average(rmse_train), average(rmse_val), average(rmse_test)))           
            
            
    
    trainPredict = b_model.predict(trainX)
    testPredict = b_model.predict(testX)
    valPredict = b_model.predict(valX)
    
    trainPredict = scalers[cont].inverse_transform(trainPredict)
    testPredict = scalers[cont].inverse_transform(testPredict)
    valPredict =scalers[cont].inverse_transform(valPredict)
    
    trainPredict = trainPredict.reshape((-1, 1))
    testPredict = testPredict.reshape((-1, 1))
    valPredict = valPredict.reshape((-1, 1))
    
    trainY = scalers[cont].inverse_transform(trainY)
    testY = scalers[cont].inverse_transform(testY)
    valY = scalers[cont].inverse_transform(valY)

    
    
    trainY = trainY.reshape((-1, 1))
    testY = testY.reshape((-1, 1))
    valY = valY.reshape((-1, 1))
    
    
    X_train = np.arange(len(trainY))
    X_val = np.arange(len(valY))
    X_test = np.arange(len(testY))
    
    
    
    #DataSet Train 2015 - 2017
    
    Data_Ini_Train=datetime.strptime('2015-01-01 01:00:00', '%Y-%m-%d  %H:%M:%S') 
    Data_Fim_Train=datetime.strptime('2017-12-31 22:00:00', '%Y-%m-%d  %H:%M:%S') 
    
    #DataSet Validação 2018
    Data_Ini_val=datetime.strptime('2018-01-01 01:00:00', '%Y-%m-%d  %H:%M:%S') 
    Data_Fim_val=datetime.strptime('2018-12-31 22:00:00', '%Y-%m-%d  %H:%M:%S') 
    
    #DataSet Teste 2019 - 2020
    Data_Ini_Test=datetime.strptime('2019-01-01 01:00:00', '%Y-%m-%d  %H:%M:%S') 
    Data_Fim_Test=datetime.strptime('2020-12-31 22:00:00', '%Y-%m-%d  %H:%M:%S') 


    tempo_treino = pd.date_range(Data_Ini_Train, Data_Fim_Train, periods = len(trainPredict))
    tempo_val = pd.date_range(Data_Ini_val, Data_Fim_val, periods = len(valPredict))
    tempo_teste = pd.date_range(Data_Ini_Test, Data_Fim_Test, periods = len(testPredict))
    
    
    
    plt.plot(tempo_treino, trainY, label = 'Real', linewidth = 0.5)
    plt.plot(tempo_treino, trainPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Treino')
    plt.legend()
    plt.savefig('Plot/treino-' + i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()
    
    
    plt.plot(tempo_val, valY, label = 'Real', linewidth = 0.5)
    plt.plot(tempo_val, valPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Validação')
    plt.legend()
    plt.savefig('Plot/val-' + i + '_sem_climaticas.png')
    plt.show()
    
    plt.plot(tempo_teste, testY, label = 'Real', linewidth = 0.5)
    plt.plot(tempo_teste, testPredict, label = 'Previsão', linewidth = 0.5)
    plt.title('Teste')
    plt.savefig('Plot/teste-' + i + '_sem_climaticas.png', bbox_inches='tight')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize = (8, 4))
    plt.plot(tempo_treino, trainY, label = 'Real', linewidth = 0.1, color = 'deepskyblue')
    plt.plot(tempo_treino, trainPredict, label = 'Treino', linewidth = 0.1, color = 'darkorange')
    plt.plot(tempo_val, valY, linewidth = 0.1, color = 'deepskyblue')
    plt.plot(tempo_val, valPredict, label = 'Validação', linewidth = 0.1, color = 'lightcoral')
    plt.plot(tempo_teste, testY, linewidth = 0.1, color = 'deepskyblue')
    plt.plot(tempo_teste, testPredict, label = 'Teste', linewidth = 0.1, color = 'seagreen')
    plt.ylim(bottom = 0)
    plt.xlim(datetime.strptime('2015-01-01 00:00:00', '%Y-%m-%d  %H:%M:%S'),
             datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d  %H:%M:%S'))
    plt.savefig('previsao_' + i + '_sem_climaticas.png')
    plt.ylabel('Geração (MWmed)')
    #plt.legend()
    plt.show()
    
    
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-loss-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('mse')
    plt.ylabel('mse')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mse-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()


    print(history.history.keys())
    plt.plot(history.history['mean_absolute_percentage_error'])
    plt.plot(history.history['val_mean_absolute_percentage_error'])
    plt.title('mean_absolute_percentage_error')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mape-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-mae-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['r_square'])
    plt.plot(history.history['val_r_square'])
    plt.title('r_square')
    plt.ylabel('r_square')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-r2-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()
    
    print(history.history.keys())
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('root_mean_squared_error')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('Época')
    plt.legend(['train', 'val'], loc='upper right')
    
    plt.savefig('Plot/Usina-rmse-'+ i + '_sem_climaticas.png', bbox_inches='tight')
    plt.show()

    cont += 1



