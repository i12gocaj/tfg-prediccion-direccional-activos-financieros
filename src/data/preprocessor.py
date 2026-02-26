import pandas as pd
import numpy as np
import os
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class DataPreprocessor:
    def __init__(self, data_dir="data/raw", processed_dir="data/processed"):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_data(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        
        # Ficheros de yfinance tienen estructura especial
        if 'history' in filename:
            try:
                # A veces viene con MultiIndex
                df = pd.read_csv(filepath, index_col=0, parse_dates=True, header=[0, 1])
                df.columns = df.columns.droplevel(1)
            except:
                # Si no, leer normal
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            # CSV normal
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
        return df

    def add_technical_indicators(self, df):
        # SMA, RSI, MACD y Bollinger
        df = df.sort_index()
        
        if 'Close' not in df.columns:
            print(f"Ojo: no hay columna 'Close'. Columnas: {df.columns}")
            return df

        # SMA
        df['SMA_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # RSI
        df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        return df

    def normalize_sentiment(self, sent_df):
        # Normaliza columnas de sentimiento a un formato comun
        if 'Sentiment_Index' in sent_df.columns:
            cols = ['Sentiment_Index']
            if 'Sentiment_Volume' in sent_df.columns:
                cols.append('Sentiment_Volume')
            return sent_df[cols]

        if 'value' in sent_df.columns:
            return sent_df[['value']].rename(columns={'value': 'Sentiment_Index'})

        if 'Close' in sent_df.columns:
            return sent_df[['Close']].rename(columns={'Close': 'Sentiment_Index'})

        if 'sentiment_score' in sent_df.columns:
            return sent_df[['sentiment_score']].rename(columns={'sentiment_score': 'Sentiment_Index'})

        # Si no encontramos una columna clara, devolvemos igual
        print(f"Warning: Sentiment column not found. Columns: {sent_df.columns}")
        return sent_df

    def merge_sentiment(self, df, sentiment_df, asset_type='crypto'):
        # Junta precios con sentimiento
        # Quitar timezone si la hay
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_localize(None)
        
        merged = df.join(sentiment_df, how='left')
        merged = merged.ffill()
        
        return merged

    def create_target(self, df, horizon=1):
        # Target: log-return a H días
        if 'Close' not in df.columns:
            return df
            
        df[f'Target_Return_{horizon}d'] = np.log(df['Close'].shift(-horizon) / df['Close'])
        df[f'Target_Direction_{horizon}d'] = (df[f'Target_Return_{horizon}d'] > 0).astype(int)
        return df

    def process_asset(self, ticker, sentiment_file, asset_type='crypto'):
        print(f"Procesando {ticker}...")
        try:
            # Cargar datos de precios
            price_file = f"{ticker.replace('-', '_')}_history.csv"
            df = self.load_data(price_file)
            
            # Añadir indicadores técnicos
            df = self.add_technical_indicators(df)
            
            # Cargar datos de sentimiento
            sent_df = self.load_data(sentiment_file)

            sent_df = self.normalize_sentiment(sent_df)
            
            # Fusionar sentimiento
            df = self.merge_sentiment(df, sent_df, asset_type)
            
            # Targets para cada horizonte
            for h in [1, 3, 7, 14, 30]:
                df = self.create_target(df, horizon=h)
            
            # Quitar NaNs
            df = df.dropna()
            
            # Guardar
            output_file = os.path.join(self.processed_dir, f"{ticker.replace('-', '_')}_processed.csv")
            df.to_csv(output_file)
            print(f"Datos procesados guardados en {output_file}")
            return True
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_all(self):
        # Procesa BTC, TSLA y NVDA
        # Procesar cripto con sentimiento especifico
        ok = True
        ok &= self.process_asset("BTC-USD", "sentiment_BTC_gdelt.csv", asset_type='crypto')

        # Procesar acciones con sentimiento de noticias
        ok &= self.process_asset("TSLA", "sentiment_TSLA_gdelt.csv", asset_type='stock')
        ok &= self.process_asset("NVDA", "sentiment_NVDA_gdelt.csv", asset_type='stock')
        if not ok:
            raise SystemExit(1)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process_all()
