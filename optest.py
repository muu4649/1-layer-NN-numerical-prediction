#ライブラリのインポート
import optuna
import keras.backend as K
from keras.datasets import fashion_mnist
from keras.layers import Convolution2D, Input, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.utils import to_categorical
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
#from mordred import descriptors, Calculator
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization,Flatten,Dropout
from keras.optimizers import SGD



#学習用データの前処理
data=np.loadtxt(".csv",delimiter=",",dtype="float")
data1=np.genfromtxt(".csv",delimiter=",",dtype="float",skip_header=1)

X=data[:,:]
Y=data1[:,:]

x_train=X[:,:]
y_train=Y[:,:]
x_test=X[:,:]
y_test=Y[:,:]

def create_model(n_layer, activation, mid_units, dropout_rate):
    model = Sequential()

    for i in range(n_layer):
        model.add(Dense(32,activation=activation))
        model.add(BatchNormalization())

#    model.add(Flatten())
    model.add(Dense(mid_units, activation=activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(units=1))

    return model


def objective(trial):

    # 調整したいハイパーパラメータの設定
    n_layer = trial.suggest_int('n_layer', 1, 5) # 追加する層を1-3から選ぶ
    mid_units = int(trial.suggest_discrete_uniform('mid_units', 50, 500, 1)) # ユニット数
    dropout_rate = trial.suggest_uniform('dropout_rate', 0, 1) # ドロップアウト率
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid']) # 活性化関数
    optimizer = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'rmsprop']) # 最適化アルゴリズム

    # 学習モデルの構築と学習の開始
    model = create_model(n_layer, activation, mid_units, dropout_rate)
    model.compile(optimizer=optimizer,
                    loss='mean_squared_error',
                    metrics=['mae'])
    history = model.fit(x_train, y_train,
                        verbose=1,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        batch_size=32)

    # 学習モデルの保存
    model_json = model.to_json()
    with open('keras_model.json', 'w') as f_model:
        f_model.write(model_json)
    model.save_weights('keras_model.hdf5')

    # 最小値探索なので
    return -np.amin(history.history['loss'])


def main():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=10)
    print('best_params')
    print(study.best_params)
    print('-1 x best_value')
    print(-study.best_value)

    print('\n --- sorted --- \n')
    sorted_best_params = sorted(study.best_params.items(), key=lambda x : x[0])
    for i, k in sorted_best_params:
        print(i + ' : ' + str(k))


if __name__ == '__main__':
    main()
