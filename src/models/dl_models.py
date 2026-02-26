import os

# Reducir el ruido de logs de TensorFlow (especialmente en macOS).
# Si se quiere ver el log completo, basta con exportar TF_CPP_MIN_LOG_LEVEL=0 antes de ejecutar.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Flatten, Dropout, Input
from sklearn.preprocessing import MinMaxScaler


tf.get_logger().setLevel("ERROR")

class DLModels:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns_ = None

    @staticmethod
    def _drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina columnas que contienen información futura (leakage).

        En este proyecto, las columnas Target_* se calculan con shift(-h) y, por tanto,
        no deben entrar nunca como variables de entrada.
        """
        if df.empty:
            return df
        leaky_cols = [c for c in df.columns if str(c).startswith("Target_")]
        if not leaky_cols:
            return df
        return df.drop(columns=leaky_cols)

    def prepare_data(self, df, window_size=30, horizon=1):
        # Prepara datos para LSTM/GRU/CNN
        data = df.copy().dropna()
        data = self._drop_leaky_columns(data)
        self.feature_columns_ = list(data.columns)
        
        target_col = 'Close'
        
        # Escalar todo
        scaled_data = self.scaler.fit_transform(data)
        self.target_scaler.fit(data[[target_col]])
        
        X, y = [], []
        target_idx = data.columns.get_loc(target_col)
        
        # Ventanas deslizantes
        for i in range(window_size, len(scaled_data) - horizon + 1):
            X.append(scaled_data[i-window_size:i])
            y.append(scaled_data[i + horizon - 1, target_idx])
            
        X, y = np.array(X), np.array(y)
        return X, y

    def build_lstm(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_gru(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(GRU(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def build_cnn(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_predict(self, train_X, train_y, test_X, model_type='lstm', epochs=10, batch_size=32):
        input_shape = (train_X.shape[1], train_X.shape[2])
        
        if model_type == 'lstm':
            model = self.build_lstm(input_shape)
        elif model_type == 'gru':
            model = self.build_gru(input_shape)
        elif model_type == 'cnn':
            model = self.build_cnn(input_shape)
        else:
            raise ValueError("Unknown model type")
            
        model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=0)
        pred = model.predict(test_X, verbose=0)
        
        # Desescalar
        pred_inv = self.target_scaler.inverse_transform(pred)
        return pred_inv

if __name__ == "__main__":
    pass
