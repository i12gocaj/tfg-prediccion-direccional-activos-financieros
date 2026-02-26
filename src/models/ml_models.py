import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class MLModels:
    def prepare_features(self, df, window_size=10, horizon=1, dropna=True):
        # Crea features de lag para RF y XGBoost
        data = df.copy()
        
        # Lags de las columnas principales
        features = ['Close', 'RSI', 'MACD', 'Sentiment_Index', 'Sentiment_Volume']
        
        for feature in features:
            if feature in data.columns:
                for i in range(1, window_size + 1):
                    data[f'{feature}_lag_{i}'] = data[feature].shift(i)
        
        # Target: log-return
        data['Target'] = np.log(data['Close'].shift(-horizon) / data['Close'])

        feature_cols = [c for c in data.columns if 'lag' in c]

        X = data[feature_cols]
        y = data['Target']
        ref_price = data['Close']

        if dropna:
            valid_mask = (
                X.notna().all(axis=1)
                & y.notna()
                & ref_price.notna()
            )
            X = X[valid_mask]
            y = y[valid_mask]
            ref_price = ref_price[valid_mask]
        
        return X, y, ref_price

    def train_predict(self, train_X, train_y, test_X, model_type='rf'):
        # Entrena y predice con RF o XGBoost
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            raise ValueError("Unknown model type")
            
        model.fit(train_X, train_y)
        pred = model.predict(test_X)
        return pred

if __name__ == "__main__":
    pass
