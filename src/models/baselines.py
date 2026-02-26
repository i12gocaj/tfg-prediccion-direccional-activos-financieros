import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class BaselineModels:
    def fit_arima(self, train_data, order=(5, 1, 0), verbose=False):
        # Ajusta ARIMA una sola vez sobre el bloque de entrenamiento.
        if verbose:
            print(f"Entrenando ARIMA{order}...")
        try:
            model = ARIMA(list(train_data), order=order)
            return model.fit()
        except Exception as e:
            print(f"ARIMA Fit Error: {e}")
            return None

    def forecast_from_state(self, model_state, steps):
        # Genera forecast a varios pasos usando el estado actual del modelo.
        try:
            forecast = model_state.forecast(steps=steps)
            return np.asarray(forecast)
        except Exception as e:
            print(f"ARIMA Forecast Error: {e}")
            return np.array([])

    def fit_predict_arima(self, train_data, test_size, order=(5, 1, 0), verbose=False):
        # Entrena ARIMA y predice
        model_fit = self.fit_arima(train_data, order=order, verbose=verbose)
        if model_fit is None:
            return np.zeros(test_size)
        forecast = self.forecast_from_state(model_fit, steps=test_size)
        if len(forecast) == 0:
            return np.zeros(test_size)
        return forecast

if __name__ == "__main__":
    # Prueba
    pass
